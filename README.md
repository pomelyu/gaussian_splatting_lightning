# gaussian_splatting_lightning
Practice gaussian splatting

## Install
```bash
# --recursive is crucial, otherwise submodule will not be cloned successfully especially diff-gaussian-rasterization
git clone https://github.com/pomelyu/gaussian_splatting_lightning.git --recursive
cd gaussian_splatting_lightning
conda env create --file environment.yml
```

## Solve submodule issues
```bash
git submodule init
git submodule update --init --recursive
pip install submodules/diff-gaussian-rasterization
```

## Scripts
```bash
# train
python -m scripts.train -c configs/train_gs.yaml

# inference
python -m scripts.render_trained_image CHECKPOINT_PLY --colmap COLMAP_DIR/sparse/0 --image COLMAP_DIR/images --down_scale=10
```

## TODO(sorted by priority)
- [] enable progressively increasing sh_degree
- [] add densification
- [] add self.spatial_lr_scale
- [] support SpareAdam
- [] figure out the usage of exposure
- [] look into cuda code
- [] enable full controls of lightning CLI
- [] connect to official GUI
- [] add depth regularization
- [] support nerf dataset
