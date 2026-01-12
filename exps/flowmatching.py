import os
import time
import torch
import numpy as np
from tqdm import tqdm
from tools.data_utils import prepare_input
from tools.dataloader import get_loaders
from models.vae.vae_vit_rope import ViTAutoencoder
from models.fm.DiT import FMWrapper, DiT
from models.fm.uniDiT import FlowMatchingWrapper, UnifiedDiT
from losses.fm import FlowMatching, MMFlowMatching
from torch.optim.lr_scheduler import CosineAnnealingLR
from tools.utils import setup_distibuted_training, setup_logger, resume_training

from tools.utils import AverageMeter
from evals.eval import eval_multimodal_fm
from evals.eval import eval_flow_matching
from models.ema import LitEma
import copy

# ----------------------------------------------------------------------------

_num_moments = 3  # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype = torch.float32  # Data type to use for initial per-tensor reduction.
_counter_dtype = torch.float64  # Data type to use for the internal counters.
_rank = 0  # Rank of the current process.
_sync_device = None  # Device to use for multiprocess communication. None = single-process.
_sync_called = False  # Has _sync() been called yet?
_counters = dict()  # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative = dict()  # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor
EMA_FREQ = 25


# ----------------------------------------------------------------------------

def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.
    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.
    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device


# ----------------------------------------------------------------------------

def flowmatching_training(rank, args):
    device = torch.device('cuda', rank)

    # Set up distributed training. -----------------------------------------------
    setup_distibuted_training(args, rank)

    sync_device = torch.device('cuda', rank) if args.n_gpus > 1 else None
    init_multiprocessing(rank=rank, sync_device=sync_device)
    torch.cuda.set_device(rank)

    # Set up logger and saving directory.
    log_, logger = setup_logger(args, rank)

    # Dataloaders
    train_loader, val_loader, _ = get_loaders(rank, copy.deepcopy(args))
    if rank == 0:
        log_(f"Loading dataset {args.data} with resolution {args.res}")
        log_(f"Loaded dataset {args.data} from folder {train_loader.dataset.path}")
        log_(f"Generating vae model")

    # Instantiate autoencoders and load weights
    autoencoder_model = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
    autoencoder_cond_model = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig).to(device)

    if rank == 0:
        log_(f"Loading pretrained vae model {args.ae_model} and {args.ae_cond_model}")
        autoencoder_cond_model_ckpt = torch.load(args.ae_cond_model)
        autoencoder_cond_model.load_state_dict(autoencoder_cond_model_ckpt)
        autoencoder_model_ckpt = torch.load(args.ae_model)
        autoencoder_model.load_state_dict(autoencoder_model_ckpt)
        # Instantiate flow matching denoising DiT
        log_(f"Generating DiT model")

    backbone = DiT(**args.dit_config, frames=args.frames)
    
    if rank == 0:
        num_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        log_(f"DiT has {num_params / 1e6:.2f}M parameters, {trainable_params / 1e6:.2f}M are trainable")

    fm_model = FMWrapper(backbone).to(device)

    ema_model = None

    opt = torch.optim.AdamW(fm_model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, T_max=args.max_iter, eta_min=0)

    last_it = 0

    if args.resume and rank == 0:  # Resume training in the same log folder
        last_it = resume_training(fm_model, opt, scheduler, logger.logdir, log_, args, last_it)

    autoencoder_model = torch.nn.parallel.DistributedDataParallel(
        autoencoder_model,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    autoencoder_cond_model = torch.nn.parallel.DistributedDataParallel(
        autoencoder_cond_model,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    fm_model = torch.nn.parallel.DistributedDataParallel(
        fm_model,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    criterion = FlowMatching(
        fm_model,
        channels=args.dit_config.in_channels,
        image_size=args.dit_config.input_size,
        loss_type='l2'
    ).to(device)

    if args.resume and rank == 0:
        criterion_ckpt = torch.load(os.path.join(logger.logdir, 'criterion_last.pth'))
        criterion.load_state_dict(criterion_ckpt)
        del criterion_ckpt

    criterion = torch.nn.parallel.DistributedDataParallel(
        criterion,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    if args.scale_lr:
        args.lr *= args.batch_size

    losses = dict()
    losses['fm_loss'] = AverageMeter()
    check = time.time()

    if ema_model == None:
        ema_model = copy.deepcopy(fm_model)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(last_it / EMA_FREQ, dtype=torch.int)
        ema_model.eval()

    autoencoder_model.eval()
    if autoencoder_cond_model is not None:
        autoencoder_cond_model.eval()
    else:
        autoencoder_cond_model = autoencoder_model
    fm_model.train()

    pbar = tqdm(total=args.max_iter, initial=last_it, dynamic_ncols=True, disable=(rank != 0))
    for it, (x, _) in enumerate(train_loader):
        it += last_it
        if it > args.max_iter:
            break
        pbar.update(1)
        z, c, _ = prepare_input(it, x, fm_model, device, autoencoder_model, args, autoencoder_cond_model)

        opt.zero_grad()
        
        loss, loss_dict = criterion(z.float(), c.float())
        
        loss.backward()
        opt.step()
        scheduler.step()

        losses['fm_loss'].update(loss.item(), 1)

        # ema model
        if it % EMA_FREQ == 0 and it > 0:
            ema(fm_model)

        if rank == 0:
            pbar.set_description(f"Loss: {losses['fm_loss'].average:.4f}")
        if it % args.log_freq == 0:
            if logger is not None and rank == 0:
                logger.scalar_summary('train/fm_loss', losses['fm_loss'].average, it)
                logger.scalar_summary('train/lr', scheduler.get_lr()[0], it)

            losses = dict()
            losses['fm_loss'] = AverageMeter()

        if it % args.eval_freq == 0 and rank == 0:
            torch.save(fm_model.module.state_dict(), os.path.join(logger.logdir, f'model_{it}.pth'))
            torch.save(fm_model.module.state_dict(), os.path.join(logger.logdir, f'model_last.pth'))
            torch.save(criterion.module.state_dict(), os.path.join(logger.logdir, f'criterion_last.pth'))
            ema.copy_to(ema_model)
            torch.save(ema_model.module.state_dict(), os.path.join(logger.logdir, f'ema_model_last.pth'))
            
            frames = args.frames + args.cond_frames
            fvd, ssim, lpips = eval_flow_matching(rank, ema_model, autoencoder_model, autoencoder_cond_model,
                                              val_loader, it, samples=args.eval_samples, logger=logger,
                                              frames=frames, cond_frames=args.cond_frames, sampling_timesteps=100)
            lpips *= 100

            if logger is not None and rank == 0:
                logger.scalar_summary('test/fvd', fvd, it)
                logger.scalar_summary('test/ssim', ssim, it)
                logger.scalar_summary('test/lpips', lpips, it)
                log_('[It %d] [FVD PRED %f] [SSIM %.2f] [LPIPS %.2f]' %
                     (it, fvd, ssim, lpips))

    pbar.close()
    if rank == 0:
        torch.save(fm_model.state_dict(), os.path.join(logger.logdir, f'net_meta.pth'))


def multimodal_flowmatching_training(rank, args):
    device = torch.device('cuda', rank)

    # Set up distributed training. -----------------------------------------------
    setup_distibuted_training(args, rank)

    sync_device = torch.device('cuda', rank) if args.n_gpus > 1 else None
    init_multiprocessing(rank=rank, sync_device=sync_device)
    torch.cuda.set_device(rank)

    # Set up logger and saving directory.
    log_, logger = setup_logger(args, rank)

    # Dataloaders
    train_loader, val_loader, _ = get_loaders(rank, copy.deepcopy(args))

    if rank == 0:
        log_(f"Loading dataset {args.data} with resolution {args.res}")
        log_(f"Loaded dataset {args.data} from folder {train_loader.dataset.path_rgb}")
        log_(f"Generating vae models for all modalities")

    # Instantiate autoencoders and load weights
    autoencoder_model_rgb = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
    autoencoder_model_depth = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
    autoencoder_cond_model_rgb = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig).to(device)
    autoencoder_cond_model_depth = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig).to(device)

    if rank == 0:
        log_(f"Loading pretrained vae model rgb {args.ae_model} and {args.ae_cond_model}, depth {args.ae_model_depth} and {args.ae_cond_model_depth}")
        autoencoder_model_rgb_ckpt = torch.load(args.ae_model)
        autoencoder_model_rgb.load_state_dict(autoencoder_model_rgb_ckpt)
        autoencoder_model_depth_ckpt = torch.load(args.ae_model_depth)
        autoencoder_model_depth.load_state_dict(autoencoder_model_depth_ckpt)
        
        autoencoder_cond_model_rgb_ckpt = torch.load(args.ae_cond_model)
        autoencoder_cond_model_rgb.load_state_dict(autoencoder_cond_model_rgb_ckpt)
        autoencoder_cond_model_depth_ckpt = torch.load(args.ae_cond_model_depth)
        autoencoder_cond_model_depth.load_state_dict(autoencoder_cond_model_depth_ckpt)
        # Define Multi-modal model based on single modalities ones.
        log_(f"Generating DiT model")
        log_(f"Using cross attention with {args.cross_attn_configs} and same noise = {args.same_noise}")

    # mm_unet = MultiModalUnet(args.unetconfig, args.unetconfig, args.ddconfig.frames, args.cross_attn_configs,
    #                              args.shared)
    # if args.fm_rgb_model != '' and args.fm_depth_model != '':
    #     mm_unet.load_single_modality_models(args.fm_rgb_model, args.fm_depth_model)

    unified_model = UnifiedDiT(**args.unified_dit_config, frames=args.frames).to(device)

    if rank == 0:
        # log number of parameters
        num_params = sum(p.numel() for p in unified_model.parameters())
        trainable_params = sum(p.numel() for p in unified_model.parameters() if p.requires_grad)
        log_(f"MultiModalDiT has {num_params / 1e6:.2f}M  parameters, {trainable_params / 1e6:.2f}M  are trainable")

    fm_model = FlowMatchingWrapper(unified_model).to(device)

    ema_model = None

    autoencoder_model_rgb = torch.nn.parallel.DistributedDataParallel(
        autoencoder_model_rgb,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    autoencoder_model_depth = torch.nn.parallel.DistributedDataParallel(
        autoencoder_model_depth,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    autoencoder_cond_model_rgb = torch.nn.parallel.DistributedDataParallel(
        autoencoder_cond_model_rgb,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)
    
    autoencoder_cond_model_depth = torch.nn.parallel.DistributedDataParallel(
        autoencoder_cond_model_depth,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    if args.scale_lr:
        args.lr *= args.batch_size

    opt = torch.optim.AdamW(fm_model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, T_max=args.max_iter, eta_min=args.lr / 100)

    last_it = 0

    if args.resume and rank == 0:  # Resume training in the same log folder
        last_it = resume_training(fm_model, opt, scheduler, logger.logdir, log_, args, last_it)

    elif args.fm_model != "" and rank == 0:  # Resume = False but finetune a pretrained model.
        log_(f"Loading pretrained model from {args.fm_model}")
        fm_model.load_state_dict(torch.load(args.fm_model))

    fm_model = torch.compile(fm_model, mode="reduce-overhead")
    fm_model = torch.nn.parallel.DistributedDataParallel(
        fm_model,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    criterion = MMFlowMatching(
        fm_model,
        channels=args.unified_dit_config.in_channels,
        image_size=args.unified_dit_config.input_size,
        same_noise=args.same_noise
    ).to(device)

    criterion = torch.nn.parallel.DistributedDataParallel(
        criterion,
        device_ids=[device],
        broadcast_buffers=False,
        find_unused_parameters=False)

    losses = dict()
    losses['fm_loss'] = AverageMeter()
    losses['fm_loss_rgb'] = AverageMeter()
    losses['fm_loss_depth'] = AverageMeter()

    start_time = time.time()

    if ema_model == None:
        ema_model = copy.deepcopy(fm_model)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(last_it / EMA_FREQ, dtype=torch.int)
        ema_model.eval()

    autoencoder_model_rgb.eval()
    autoencoder_model_depth.eval()
    if autoencoder_cond_model_rgb is not None:
        autoencoder_cond_model_rgb.eval()
        autoencoder_cond_model_depth.eval()
    else:
        autoencoder_cond_model_rgb = autoencoder_model_rgb
        autoencoder_cond_model_depth = autoencoder_model_depth

    fm_model.train()

    pbar = tqdm(total=args.max_iter, initial=last_it, dynamic_ncols=True, disable=(rank != 0))
    for it, (x_rgb, x_depth, _) in enumerate(train_loader):
        it += last_it
        if it > args.max_iter:
            break
        pbar.update(1)
        z_rgb, c_rgb, p = prepare_input(it, x_rgb, fm_model, device, autoencoder_model_rgb, args,
                                        autoencoder_cond_model_rgb)
        rgb_cond_active = p < args.cond_prob
        if args.modality_guidance:
            p = None  # resample p for depth modality if modality guidance is enabled
        z_depth, c_depth, p = prepare_input(it, x_depth, fm_model, device, autoencoder_model_depth, args,
                                            autoencoder_cond_model_depth, p)
        depth_cond_active = p < args.cond_prob

        # if both modalities are not conditioned, we need to condition the model with at least one of them (rgb)
        if not rgb_cond_active and not depth_cond_active:
            save_prob = args.cond_prob
            args.cond_prob = 2  # be sure that the model is conditioned
            z_rgb, c_rgb, _ = prepare_input(it, x_rgb, fm_model, device, autoencoder_model_rgb, args,
                                            autoencoder_cond_model_rgb)
            args.cond_prob = save_prob  # restore the original value

        b_size = z_rgb.shape[0]
        
        is_sync = np.random.rand() < 0.5
        if is_sync:
            # t_rgb = t_depth
            t_rand = torch.sigmoid(torch.rand((b_size,), device=device))
            t_rgb = t_rand
            t_depth = t_rand
            use_same_noise = True
        else:
            # Independent sampling
            t_rgb = torch.sigmoid(torch.rand((b_size,), device=device))
            t_depth = torch.sigmoid(torch.rand((b_size,), device=device))
            use_same_noise = False
            
        opt.zero_grad()

        loss_rgb, loss_depth, loss_dict = criterion(z_rgb, z_depth, c_rgb, c_depth, t_rgb, t_depth, same_noise=use_same_noise)

        loss = loss_rgb + loss_depth

        loss.backward()
        opt.step()
        scheduler.step()

        losses['fm_loss_rgb'].update(loss_rgb.item(), 1)
        losses['fm_loss_depth'].update(loss_depth.item(), 1)
        losses['fm_loss'].update(loss.item(), 1)

        # ema model
        if it % EMA_FREQ == 0 and it > 0:
            ema(fm_model)

        if rank == 0:
            pbar.set_description(f"Loss: {losses['fm_loss'].average:.4f} | R: {losses['fm_loss_rgb'].average:.4f} | D: {losses['fm_loss_depth'].average:.4f}")
        if it % args.log_freq == 0:
            if logger is not None and rank == 0:
                logger.scalar_summary('train/fm_loss', losses['fm_loss'].average, it)
                logger.scalar_summary('train/fm_loss_rgb', losses['fm_loss_rgb'].average, it)
                logger.scalar_summary('train/fm_loss_depth', losses['fm_loss_depth'].average, it)
                logger.scalar_summary('train/lr', scheduler.get_lr()[0], it)

            losses = dict()
            losses['fm_loss'] = AverageMeter()
            losses['fm_loss_rgb'] = AverageMeter()
            losses['fm_loss_depth'] = AverageMeter()

        if it % args.eval_freq == 0 and rank == 0:
            torch.save(fm_model.module.state_dict(), os.path.join(logger.logdir, f'model_{it}.pth'))
            ema.copy_to(ema_model)
            torch.save(ema_model.module.state_dict(), os.path.join(logger.logdir, f'ema_model_{it}.pth'))

            frames = args.frames + args.cond_frames
            fvd_rgb, fvd_depth, ssim_rgb, ssim_depth, lpips_rgb, lpips_depth, l2 = eval_multimodal_fm(
                rank,
                ema_model=ema_model,
                ae_rgb=autoencoder_model_rgb, ae_depth=autoencoder_model_depth,
                ae_cond_rgb=autoencoder_cond_model_rgb, ae_cond_depth=autoencoder_cond_model_depth,
                loader=val_loader, it=it, samples=args.eval_samples, logger=logger, frames=frames,
                cond_frames=args.cond_frames, same_noise=args.same_noise
            )

            lpips_rgb *= 1000
            lpips_depth *= 1000
            l2 *= 100

            if logger is not None and rank == 0:
                logger.scalar_summary('test/fvd_rgb', fvd_rgb, it)
                logger.scalar_summary('test/fvd_depth', fvd_depth, it)
                logger.scalar_summary('test/ssim_rgb', ssim_rgb, it)
                logger.scalar_summary('test/ssim_depth', ssim_depth, it)
                logger.scalar_summary('test/lpips_rgb', lpips_rgb, it)
                logger.scalar_summary('test/lpips_depth', lpips_depth, it)
                logger.scalar_summary('test/l2', l2, it)

                log_('[It %d] [FVD_RGB %f] [FVD_DEPTH %f] [SSIM RGB %.2f - DEPTH %.2f] [LPIPS RGB %.2f - DEPTH %.2f] [L2 %.4f]' %
                    (it, fvd_rgb, fvd_depth, ssim_rgb, ssim_depth, lpips_rgb, lpips_depth, l2))

    pbar.close()
    if rank == 0:
        torch.save(fm_model.state_dict(), os.path.join(logger.logdir, f'net_meta.pth'))
