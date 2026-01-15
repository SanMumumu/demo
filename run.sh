# vae (ideally one per modality)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --config config_run/train_vae/vae_city_rgb.yaml --num_workers 24 --output=./Archived_AE

CUDA_VISIBLE_DEVICES=2,3,4,5 python3 main.py --config config_run/train_vae/vae_city_depth.yaml --output=./debug1

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --config config_run/train_fm/fm_city_rgb.yaml --num_workers 8 --output=./debug1

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py --config config_run/train_fm/fm_city_rgb.yaml --output=./debug1

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py --config config_run/train_fm/fm_city_rgb.yaml --output=./debug1

# single modality  model
python3 main.py --config configs/run/train/ddpm_city_rgb.yaml


tensorboard --logdir=/mnt/data/wangsen/demo/results --port=7004