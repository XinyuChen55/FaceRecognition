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

# download model checkpoint
mkdir -p checkpoints/retinanet_r50_fpn_1x_coco
wget -O checkpoints/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth \
https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth

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

# install Kaggle CLI for dataset download
pip install kaggle

### 1: Generate API Token
1. Sign in to Kaggle.
2. Go to **Setting**.
3. Click **Create Legacy API Key**.
4. A file named `kaggle.json` will be downloaded.

### 2: Place `kaggle.json` in the Correct Directory
mkdir ~\.kaggle
move kaggle.json ~\.kaggle\

# download LFW dataset
mkdir -p data/lfw_raw
kaggle datasets download -d jessicali9530/lfw-dataset -p data/lfw_raw
unzip data/lfw_raw/lfw-dataset.zip -d data/lfw

# download CelebA dataset
mkdir -p data/celebA_raw
kaggle datasets download -d jessicali9530/celeba-dataset -p data/celebA_raw
unzip data/celebA_raw/celeba-dataset.zip -d data/celebA

# download MMHuman3D Body Model
mkdir -p data/body_models
mkdir -p data/body_models/smpl

Register and download the SMPL model files from the official sources: https://smpl.is.tue.mpg.de/index.html and place them in data/body_models/smpl/ 
For example, `mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl SMPL_NEUTRAL.pkl`

wget -O data/body_models/J_regressor_extra.npy "https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1" 

wget -O data/body_models/J_regressor_h36m.npy "https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi"

wget -O data/body_models/smpl_mean_params.npz "https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4"

# checkpoint for MMHuman3D
mkdir -p checkpoints/MMHuman3D
wget -O checkpoints/MMHuman3D/resnet50_hmr_pw3d.pth https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hmr/resnet50_hmr_pw3d-04f40f58_20211201.pth?versionId=CAEQHhiBgMD6zJfR6xciIDE0ODQ3OGM2OWJjMTRlNmQ5Y2ZjMWZhMzRkOTFiZDFm

# download WIDER FACE Dataset
Go to: `http://shuoyang1213.me/WIDERFACE/`

Download the following files:
- **WIDER Face Training Images**
- **WIDER Face Validation Images**
- **WIDER Face Tesing Images**
- **Face annotations**

mkdir -p data/widerface
mv /path/to/your/downloads/WIDER_*.zip data/widerface/
mv /path/to/your/downloads/wider_face_split*.zip data/widerface/
cd data/widerface
unzip -q WIDER_train.zip
unzip -q WIDER_val.zip
unzip -q WIDER_test.zip
unzip -q wider_face_split.zip

# download MMPose
There will be conflicts about PyTorch version, etc. so we need to create new env

conda create -n mmpose python=3.10 -y
conda activate mmpose

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install numpy==1.26.4
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-python==4.8.1.78

pip install -U pip setuptools wheel
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmdet==3.2.0

cd third_party
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .