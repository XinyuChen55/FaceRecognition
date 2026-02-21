# Create conda environment
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab

# Install ffmpeg
conda install ffmpeg

# Install PyTorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# install PyTorch3D
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install .
cd ..

# install OpenCV & Numpy
conda install -y -c conda-forge opencv numpy=1.23.5

# Install mmcv-full
pip install -U openmim
mim install "mmcv-full==1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# install mmdetection
mkdir -p third_party
git clone --branch v2.25.1 --depth 1 https://github.com/open-mmlab/mmdetection.git third_party/mmdetection
pip install -e third_party/mmdetection

# Install mmhuman3d
git clone https://github.com/open-mmlab/mmhuman3d.git third_party/mmhuman3d
pip install -e thrid_party/mmhuman3d

# verify installation
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, 'torch.cuda', torch.version.cuda, 'cuda_avail', torch.cuda.is_available())"
python -c "import cv2; print('opencv', cv2.__version__)"
python -c "import mmcv; print('mmcv', mmcv.__version__)"
python -c "import mmdet; print('mmdet', mmdet.__version__)"
python -c "import mmhuman3d; print('mmhuman3d', mmhuman3d.__version__)"
python -c "import numpy as np; print('numpy', np.__version__)"