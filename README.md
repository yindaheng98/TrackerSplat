# InstantSplatFlow

Fast volumetric video reconstruction, just like InstantSplat

## Prerequisites

* [Pytorch](https://pytorch.org/) (v2.4 or higher recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, should match with PyTorch version)
* [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation) (mmsegmentation==1.2.2 mmcv==2.1.0)

Install dependencies:
```sh
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --upgrade git+https://github.com/yindaheng98/InstantSplat.git@main
pip install --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main
pip install --upgrade git+https://github.com/facebookresearch/co-tracker.git@main
pip install --upgrade git+https://github.com/facebookresearch/sam2@main
pip install -U openmim ftfy regex && python -m mim install -U mmsegmentation==1.2.2 mmcv==2.1.0
pip install -U xformers==0.0.12 --no-deps
pip install -U scikit-learn taichi einops einshape timm tifffile triton
conda install conda-forge::colmap
```

## Install (pip, build from source)

```shell
pip install --upgrade git+https://github.com/yindaheng98/InstantSplatStream.git@main
```

## Install (Development)

```shell
git clone https://github.com/yindaheng98/InstantSplatStream --recursive
cd InstantSplatStream
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/InstantSplat.git@main
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main
pip install --target . --upgrade .
```

## Download pth

Depth Anything V2
```sh
wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

DUST3R
```sh
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

dot
```shell
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_tapir.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_plus_bootstapir.pth
```

cotracker
```shell
wget -P checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

dinov2
```shell
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/ssl_default_config.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vits14_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitl14_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitb14_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitg14_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vits14_reg4_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitl14_reg4_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitb14_reg4_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitg14_reg4_pretrain.yaml
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth
wget -P configs/dinov2 https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_ms_config.py
wget -P configs/dinov2 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_ms_config.py
wget -P configs/dinov2 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_ms_config.py
wget -P configs/dinov2 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_ms_config.py
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_ms_head.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_ms_head.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_ms_head.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_ms_head.pth
```

sam2
```shell
wget -P configs/sam2.1 https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
wget -P configs/sam2.1 https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml
wget -P configs/sam2.1 https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml
wget -P configs/sam2.1 https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Quick Start

### Prepare Datasets

Download datasets and extract them into `./data`:

* [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0)
* [st-nerf dataset](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xliufe_connect_ust_hk/EjqArjZxmmtDplj_IrwlUq0BMUyG69zr5YqXFBxgku4rRQ?e=n2fSBs)
* [Meet Room dataset](https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_)
* [Dynamic 3D Gaussians dataset](https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip)

For Neural 3D Video dataset:
```
data
|-coffee_martini
  |-poses_bounds.npy
  |-cam00.mp4
  |-cam01.mp4
  |-cam02.mp4
  |-cam......
|-cook_spinach
  |-poses_bounds.npy
  |-cam00.mp4
  |-cam01.mp4
  |-cam02.mp4
  |-cam......
......
```

For st-nerf dataset:
```
data
|-boxing
  |-boxing.zip
|-taekwondo
  |-taekwondo.zip
|-walking
  |-walking.zip
......
```

For Meet Room dataset:
```
data
|-discussion
  |-poses_bounds.npy
  |-cam_0.mp4
  |-cam_1.mp4
  |-cam_2.mp4
  |-cam......
|-stepin
  |-poses_bounds.npy
  |-cam_0.mp4
  |-cam_1.mp4
  |-cam_2.mp4
  |-cam......
......
```

For Dynamic 3D Gaussians dataset:
```
data
|-basketball
  |-ims
  |-seg
  |-init_pt_cld.npz
  |-test_meta.json
  |-train_meta.json
|-boxes
  |-ims
  |-seg
  |-......
......
```

Run scripts to extract n3dv-like datasets (n3dv, st-nerf and meet room) to proper format:
```sh
./tools/n3dv_extract.sh
```

Run scripts to extract and initialize camera poses for all datasets:
```sh
./tools/init_dataset.sh
```

### Run the experiment

Experiment on all our methods and baselines
```sh
./tools/motionestimation.sh
```

Then you can see the quality (PANR, SSIM, LIPIPS) of each training step in output folder: `output/<name of dataset>/<name of method>/frame2/log/iteration_1000/log.csv`

### Long video experiment

TBD