# GALAv2

## Overview


## Installation Guide
### Clone Repository


### Environment Setup
```bash
conda create -n galav2 python=3.10
conda activate galav2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```

### Project Dependencies
```bash
pip install -r requirements.txt
micromamba install -c conda-forge gxx=11.4.0
```

### Submodules
```bash
# local install
pip install -e submodules/gsplat --no-build-isolation
pip install -e submodules/simple-knn --no-build-isolation

# install from github (recommend)
pip install git+https://github.com/nerfstudio-project/gsplat.git
pip install git+https://github.com/camenduru/simple-knn
```

## Preprocessing Requirements
### Install SAM for segmentation
```bash
# vanilla SAM
# pip install git+https://github.com/facebookresearch/segment-anything.git

# langsplat version
git clone https://github.com/minghanqin/segment-anything-langsplat.git
cd segment-anything
pip install -e .

# download checkpoints
cd ../../ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Huggingface for DINOv3 (optional)
```bash
pip install -U git+https://github.com/huggingface/transformers.git
```

### Install for DINOtxt (optional)
```bash
pip install -r requirements_dinotxt.txt
```

### Install CLIP (optional)
```bash
pip install open-clip-torch
```

## Dataset Preparation
### Input Dataset
The dataset follows a structured format where each 3D scene is organized as follows:
```
lerf_ovs/
└── scene_name/           # Name of the specific scene (e.g., teatime)
    ├── distorted/        
    ├── images/           # Contains the original, unprocessed scene images
    ├── language_features/ # Pre-extracted language embeddings
    │   ├── frame_00001_f.npy
    │   └── frame_00001_s.npy
    │   ├── ...
    ├── sparse/0/      
    │   ├── test.txt     # Testing image list
    │   ├── cameras.bin 
    │   ├── images.bin
    │   └── points3D.bin 
    ├── stereo/         
```
Notes:
- Language features are pre-extracted and stored as 512-dimensional vectors
- For detailed information about feature levels and language feature extraction methodology, please refer to the [LangSplat repository](https://github.com/minghanqin/LangSplat). 

### Output Directory Structure
The pre-trained RGB model outputs are organized as follows:
```
output/
└── dataset_name/
    └── scene_name/
        ├── point_cloud/
        │   └── iteration_30000/
        │       └── point_cloud.ply      # Point cloud at 30K iterations
        ├── cameras.json                 
        ├── cfg_args                     
        ├── chkpnt30000.pth             # Model checkpoint at 30K iterations
        └── input.ply                    

```


## Usage
### Prerequisites
#### Preprocessing with SAM and feature extractor
```bash
# Train gaussian model
python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs/figurines --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder dinov3 --empty_bg
```


#### 1. Train and Render RGB Gaussian Model
```bash
# Train gaussian model
python train.py -s $DATA_SOURCE_PATH -m $MODEL_OUTPUT_PATH --iterations 30000
```

