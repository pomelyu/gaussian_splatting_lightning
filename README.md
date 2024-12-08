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
