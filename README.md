# InstantSplatFlow

Fast volumetric video reconstruction, just like InstantSplat

## Install

### Requirements

Install Pytorch and torchvision following the official guideline: [pytorch.org](https://pytorch.org/)

Install Pillow, numpy and tqdm
```sh
pip install Pillow numpy tqdm
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --upgrade git+https://github.com/yindaheng98/InstantSplat.git@main
pip install --upgrade git+https://github.com/facebookresearch/co-tracker.git@main
pip install --upgrade git+https://github.com/facebookresearch/dinov2@main xformers --index-url https://download.pytorch.org/whl/cu121 --no-deps
```

### Pip Install

```shell
pip install --upgrade git+https://github.com/yindaheng98/InstantSplatStream.git@main
```

### Local Install

```shell
git clone https://github.com/yindaheng98/InstantSplatStream --recursive
cd InstantSplatStream
pip install --target . --upgrade .
```

## Download pth

```shell
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_tapir.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_plus_bootstapir.pth
wget -P checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

```shell
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vits14_reg4_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitl14_reg4_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitb14_reg4_pretrain.yaml
wget -P configs/dinov2 https://raw.githubusercontent.com/facebookresearch/dinov2/refs/heads/main/dinov2/configs/eval/vitg14_reg4_pretrain.yaml
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth
```