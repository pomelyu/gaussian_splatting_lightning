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
- [x] use pycolmap to load the COLMAP data
- [x] break down official inference code(remove scene)
- [x] replace official GaussianModel for inference
- [x] setup `environment.yml` for pytorch2.0.1 and cuda11.8 
- [x] implement basic training without densification by `pytorch_lightning`, `mlflow` and `mlconfig`(`omegaconfig`)
- [x] add densification
- [x] support `GSWarmUpExponentialDecayScheduler`
- [x] try [gsplat](https://github.com/nerfstudio-project/gsplat) in `dev-gsplat` [branch](https://github.com/pomelyu/gaussian_splatting_lightning/tree/dev-gsplat)
- [x] look into forward cuda code
- [x] implement runnable python rasterization function
- [ ] implement runnable cuda rasterization function
- [ ] look into backward cuda code
- [ ] figure out the usage of exposure
- [ ] support SpareAdam
- [ ] enable full controls of lightning CLI
- [ ] connect to official GUI
- [ ] add depth regularization
- [ ] support nerf dataset
