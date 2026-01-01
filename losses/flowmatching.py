import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from tqdm import tqdm
from functools import partial
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class FlowMatching(nn.Module):
    def __init__(self,
                 model,
                 timesteps=1000,        # timesteps used for config compatibility; we use continuous time in practice
                 sampling_timesteps=10, # FM requires fewer steps, e.g., 10-50
                 loss_type="l2",
                 image_size=32,
                 channels=4,
                 sigma_min=1e-5,        # Avoid numerical instability at t=0 or t=1
                 **kwargs
                 ):
        super().__init__()
        self.model = model
        self.channels = channels
        self.loss_type = loss_type
        
        # Handle latent dimension size retrieval
        try:
            self.image_size = model.module.diffusion_model.ae_emb_dim
        except AttributeError:
            # Handle property access path for multi-modal models
            try:
                self.image_size = model.module.diffusion_model.diff_rgb.diffusion_model.ae_emb_dim
            except:
                self.image_size = image_size # Fallback

        self.sigma_min = sigma_min
        self.sampling_timesteps = sampling_timesteps
        
        # Scale factor to map t in [0,1] to the model's preferred [0, 1000] range
        self.time_scale_factor = 1000.0 

    def get_loss(self, pred, target):
        if self.loss_type == 'l1':
            return (target - pred).abs().mean()
        elif self.loss_type == 'l2':
            return torch.nn.functional.mse_loss(target, pred)
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

    def forward(self, x, cond=None, *args, **kwargs):
        """
        x: ground truth data (x_1)
        """
        b, device = x.shape[0], x.device
        
        # 1. Logit-Normal sampling for time t ~ U[0, 1]
        t = torch.sigmoid(torch.randn((b,), device=device)) # torch.rand((b,), device=device)
        
        # 2. Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x)
        x_1 = x # Target data
        
        # 3. Linear interpolation (Optimal Transport Probability Path)
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1 
        # Simplified: x_t = (1-t)x_0 + t*x_1 (assuming sigma_min -> 0)
        # We use the version with sigma_min for numerical stability
        t_expanded = t.view(b, *([1] * (len(x.shape) - 1)))
        mu_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        
        # 4. Calculate target velocity vector (Flow Target)
        # For OT-CFM (Optimal Transport Conditional Flow Matching), the target is: x_1 - x_0
        target_v = x_1 - x_0

        # 5. Model predicts velocity v_t
        # Note: UNet Time Embeddings are usually designed for 0-1000 integers or large floats.
        # Passing 0-1 directly can cause Positional Embedding collapse, so we scale by 1000.
        model_out = self.model(mu_t, cond, t * self.time_scale_factor)
        
        # 6. Compute Loss
        loss = self.get_loss(model_out, target_v)
        
        loss_dict = {'train/loss': loss.item()}
        return loss, loss_dict

    @torch.no_grad()
    def sample(self, batch_size=16, cond=None, return_intermediates=False):
        # Euler ODE Solver
        device = next(self.model.parameters()).device
        
        # Initial noise x_0
        shape = (batch_size, self.channels, self.image_size)
        x = torch.randn(shape, device=device)
        
        dt = 1.0 / self.sampling_timesteps
        intermediates = [x]
        
        for i in range(self.sampling_timesteps):
            t_value = i / self.sampling_timesteps
            t = torch.full((batch_size,), t_value, device=device)
            
            # Predict velocity field v
            v_pred = self.model(x, cond, t * self.time_scale_factor)
            
            # Euler step: x_{t+dt} = x_t + v * dt
            x = x + v_pred * dt
            
            if return_intermediates:
                intermediates.append(x)
                
        if return_intermediates:
            return x, intermediates
        return x

class MMFlowMatching(FlowMatching):
    def __init__(self, *args, **kwargs):
        self.same_noise = kwargs.pop('same_noise', False)
        super().__init__(*args, **kwargs)

    def forward(self, x_rgb, x_depth, cond_rgb=None, cond_depth=None, *args, **kwargs):
        b, device = x_rgb.shape[0], x_rgb.device
        
        # 1. Logit-Normal sampling for time t
        t = torch.sigmoid(torch.randn((b,), device=device)) # torch.rand((b,), device=device)
        t_expanded = t.view(b, *([1] * (len(x_rgb.shape) - 1)))

        # 2. Sample noise x_0
        x_0_rgb = torch.randn_like(x_rgb)
        if self.same_noise:
            x_0_depth = x_0_rgb
        else:
            x_0_depth = torch.randn_like(x_depth)
            
        x_1_rgb = x_rgb
        x_1_depth = x_depth

        # 3. Interpolate to get x_t
        x_t_rgb = (1 - t_expanded) * x_0_rgb + t_expanded * x_1_rgb
        x_t_depth = (1 - t_expanded) * x_0_depth + t_expanded * x_1_depth

        # 4. Define target velocity v
        target_v_rgb = x_1_rgb - x_0_rgb
        target_v_depth = x_1_depth - x_0_depth

        # 5. Model prediction
        # Assumes model forward accepts (x_rgb, x_depth, cond_rgb, cond_depth, t)
        v_pred_rgb, v_pred_depth = self.model(x_t_rgb, x_t_depth, cond_rgb, cond_depth, t * self.time_scale_factor)

        # 6. Loss Calculation
        loss_rgb = self.get_loss(v_pred_rgb, target_v_rgb)
        loss_depth = self.get_loss(v_pred_depth, target_v_depth)
        
        loss_dict = {
            'train/loss_rgb': loss_rgb.item(),
            'train/loss_depth': loss_depth.item()
        }
        
        return loss_rgb, loss_depth, loss_dict

    @torch.no_grad()
    def sample(self, batch_size=16, cond_rgb=None, cond_depth=None, return_intermediates=False):
        device = next(self.model.parameters()).device
        shape = (batch_size, self.channels, self.image_size)
        
        # Initial noise
        x_rgb = torch.randn(shape, device=device)
        if self.same_noise:
            x_depth = x_rgb.clone()
        else:
            x_depth = torch.randn(shape, device=device)
            
        dt = 1.0 / self.sampling_timesteps
        
        for i in range(self.sampling_timesteps):
            t_value = i / self.sampling_timesteps
            t = torch.full((batch_size,), t_value, device=device)
            
            # Predict joint velocity fields
            v_pred_rgb, v_pred_depth = self.model(x_rgb, x_depth, cond_rgb, cond_depth, t * self.time_scale_factor)
            
            # Euler integration step
            x_rgb = x_rgb + v_pred_rgb * dt
            x_depth = x_depth + v_pred_depth * dt
            
        return x_rgb, x_depth