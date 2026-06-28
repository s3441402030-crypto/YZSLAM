# YZ-SLAM: Semantic-Geometric Dynamic Feature Filtering for Robust RGB-D SLAM in Dynamic Indoor Scenes

This repository provides the source code and reproduction instructions for **YZ-SLAM**, a semantic-geometric RGB-D SLAM framework for robust localization in dynamic indoor scenes.

YZ-SLAM is built upon a feature-based RGB-D SLAM backbone and integrates a YOLO-based semantic segmentation front-end with geometric dynamic feature filtering. The semantic front-end provides candidate dynamic regions, while depth-aware mask refinement and feature-level geometric verification are used to reduce the influence of dynamic objects before pose optimization.

---

## 1. Overview

Dynamic objects often violate the static-scene assumption of conventional visual SLAM systems, leading to incorrect feature associations, pose drift, and degraded map quality. YZ-SLAM addresses this problem by combining semantic priors and RGB-D geometric constraints.

The main pipeline includes:

1. RGB-D image input.
2. YOLO segmentation-based candidate dynamic object detection.
3. Depth-aware mask refinement.
4. Dynamic feature filtering before pose optimization.
5. RGB-D SLAM tracking and mapping.
6. Qualitative dense mapping validation.

---

## 2. Main Features

* Integration of YOLO segmentation with an RGB-D SLAM framework.
* Semantic candidate dynamic region generation.
* Depth-aware mask refinement for reducing boundary leakage.
* Feature-level dynamic point filtering before pose estimation.
* Support for TUM RGB-D and Bonn RGB-D Dynamic Dataset experiments.
* Source code organization for academic reproduction and further development.

---

## 3. Repository Structure

```text
YZSLAM/
├── Examples/              # Example running scripts and configuration files
├── Thirdparty/            # Third-party dependencies except LibTorch
├── Vocabulary/            # ORB vocabulary package
├── evaluation/            # Evaluation-related scripts or files
├── include/               # Header files
├── src/                   # Source files
├── best.torchscript       # YOLO segmentation TorchScript model
├── voc.names              # VOC/SBD class names
├── CMakeLists.txt         # CMake configuration
├── Seqtest                # Example shell script
├── .gitignore
└── README.md
```

---

## 4. External Resources and Download Links

### 4.1 LibTorch

LibTorch is not included in this repository because of its large file size. Please download the appropriate LibTorch package according to your CUDA version and system environment.

Official PyTorch / LibTorch download page:

```text
https://pytorch.org/get-started/locally/
```

Official C++ / LibTorch installation documentation:

```text
https://docs.pytorch.org/cppdocs/installing.html
```

After downloading and extracting LibTorch, place it under:

```text
Thirdparty/libtorch/
```

The expected directory structure is:

```text
Thirdparty/
└── libtorch/
    ├── include/
    ├── lib/
    └── share/
```

Please make sure that the LibTorch version is compatible with the CUDA version and NVIDIA driver used on your system.

---

### 4.2 TUM RGB-D Dataset

The TUM RGB-D Dataset can be downloaded from the official website:

```text
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
```

Dataset homepage:

```text
https://cvg.cit.tum.de/data/datasets/rgbd-dataset
```

---

### 4.3 Bonn RGB-D Dynamic Dataset

The Bonn RGB-D Dynamic Dataset can be downloaded from the official website:

```text
https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html
```

---

## 5. External Files

### 5.1 ORB Vocabulary

The original ORB vocabulary file `ORBvoc.txt` is large, so this repository provides a compressed vocabulary file:

```text
Vocabulary/ORBvoc.zip
```

Before compiling or running the system, extract it:

```bash
cd Vocabulary
unzip ORBvoc.zip
cd ..
```

After extraction, the expected structure is:

```text
Vocabulary/
├── ORBvoc.zip
└── ORBvoc.txt
```

`ORBvoc.txt` is required by the RGB-D SLAM system.

---

### 5.2 YOLO Segmentation Model

The pretrained YOLO segmentation model is provided as:

```text
best.torchscript
```

The class name file is:

```text
voc.names
```

The current model follows the VOC/SBD 20-class segmentation setting. In the RGB-D SLAM experiments, the `person` class is used as the main dynamic-object semantic prior.

---

## 6. Reproducibility Environment

The following environment is the reference environment used for developing and testing this project. Exact hardware matching is not strictly required, but a CUDA-capable NVIDIA GPU is recommended for YOLO segmentation inference.

### 6.1 Reference Hardware

The experiments in the manuscript were mainly conducted on the following workstation:

```text
GPU: NVIDIA GeForce RTX 4090
CPU: Intel Core i9-13900KF
RAM: 128 GB DDR5
Operating System: Ubuntu 20.04
Input RGB-D resolution: 640 x 480
```

A lower-end GPU can also be used for functional testing, but the real-time performance may be lower than the reported experimental results.

---

### 6.2 SLAM Runtime Environment

The RGB-D SLAM system is implemented in C++ and depends on ORB-SLAM3-style third-party libraries, OpenCV, Pangolin, Eigen, CUDA, and LibTorch.

Recommended runtime environment:

```text
Operating System: Ubuntu 20.04 / Ubuntu 22.04
Compiler: GCC/G++ 9 or later
CMake: >= 3.10
OpenCV: 4.x
Eigen3: 3.x
Pangolin: compatible with ORB-SLAM3
CUDA: CUDA 11.x / CUDA 12.x
LibTorch: C++ distribution compatible with the local CUDA version
```

Reference development environment:

```text
Ubuntu 20.04
OpenCV 4.5
Eigen3
CUDA-enabled LibTorch
```

LibTorch is not included in this repository because of its large file size. Please download the C++ LibTorch package manually and place it under:

```text
Thirdparty/libtorch/
```

Expected structure:

```text
Thirdparty/
└── libtorch/
    ├── include/
    ├── lib/
    └── share/
```

Please make sure that the LibTorch CUDA version is compatible with your local NVIDIA driver and CUDA runtime.

---

### 6.3 YOLO Training Environment

The YOLO segmentation model was trained in a Python environment. The following environment is recommended for reproducing or fine-tuning the semantic segmentation model.

Create a new Conda environment:

```bash
conda create -n yzslam_yolo python=3.10 -y
conda activate yzslam_yolo
```

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Install common training dependencies:

```bash
pip install ultralytics opencv-python numpy matplotlib pandas tqdm pyyaml scipy
```

Recommended YOLO training configuration:

```text
Dataset: VOC + SBD
Number of classes: 20
Input size: 640 x 640
Training epochs: 200
Batch size: 32
Train/validation split: 8:2
Model format for SLAM deployment: TorchScript
```

The pretrained TorchScript model used by the SLAM system is provided as:

```text
best.torchscript
```

The corresponding class-name file is:

```text
voc.names
```

The current deployment model follows the VOC/SBD 20-class setting. In the RGB-D SLAM experiments, the `person` category is used as the main dynamic-object semantic prior.

---

### 6.4 Python Environment for Evaluation Scripts

Some evaluation scripts may require a lightweight Python environment. A typical setup is:

```bash
conda create -n yzslam_eval python=3.8 -y
conda activate yzslam_eval
pip install numpy matplotlib pandas scipy evo
```

The `evo` package can be used for APE/RPE trajectory evaluation if trajectory files are exported in a compatible format.

---

### 6.5 Important Environment Notes

Before building or running the system, please check the following items:

```text
1. Vocabulary/ORBvoc.txt has been extracted from Vocabulary/ORBvoc.zip.
2. Thirdparty/libtorch/ has been placed correctly.
3. best.torchscript exists in the repository root directory.
4. voc.names exists in the repository root directory.
5. OpenCV, Pangolin, Eigen3, CUDA, and LibTorch are correctly installed.
6. The dataset path and association file path are correctly set.
7. The YOLO model path in the source code is consistent with your local repository path.
```

If the YOLO model path is hard-coded in `src/YoloDetect.cpp`, please modify it before compilation. For example:

```cpp
static const std::string MODEL_PATH = "/your/path/to/YZSLAM/best.torchscript";
static const std::string CLASS_PATH = "/your/path/to/YZSLAM/voc.names";
```

The exact dependency versions do not need to be identical to the reference environment, but CUDA, LibTorch, and the NVIDIA driver must be mutually compatible.

---

## 7. Build Instructions

First, clone the repository:

```bash
git clone https://github.com/s3441402030-crypto/YZSLAM.git
cd YZSLAM
```

Then prepare the ORB vocabulary:

```bash
cd Vocabulary
unzip ORBvoc.zip
cd ..
```

Place LibTorch under:

```text
Thirdparty/libtorch/
```

Then build the project:

```bash
mkdir build
cd build
cmake ..
make -j
cd ..
```

If CMake cannot find LibTorch, please check the LibTorch path in `CMakeLists.txt` and make sure it points to:

```text
Thirdparty/libtorch/
```

---

## 8. Dataset Preparation

The datasets are **not included** in this repository. Please download the datasets manually and place them under the local `datasets/` folder in the repository root directory.

Recommended local dataset structure:

```text
YZSLAM/
└── datasets/
    ├── TUM/
    │   ├── rgbd_dataset_freiburg1_xyz/
    │   ├── rgbd_dataset_freiburg2_desk/
    │   ├── rgbd_dataset_freiburg3_sitting_static/
    │   ├── rgbd_dataset_freiburg3_sitting_xyz/
    │   ├── rgbd_dataset_freiburg3_walking_static/
    │   └── rgbd_dataset_freiburg3_walking_rpy/
    └── Bonn/
        ├── rgbd_bonn_person_tracking/
        └── rgbd_bonn_crowd/
```

The `datasets/` folder is only used locally and should not be uploaded to GitHub.

Please make sure `.gitignore` contains:

```text
datasets/
```

---

## 9. Running TUM RGB-D Sequences

Please place TUM RGB-D sequences under:

```text
datasets/TUM/
```

Example TUM sequence structure:

```text
datasets/
└── TUM/
    └── rgbd_dataset_freiburg3_walking_rpy/
        ├── rgb/
        ├── depth/
        ├── groundtruth.txt
        └── associations.txt
```

If `associations.txt` is not provided, it should be generated according to the RGB and depth timestamps.

The general command format for TUM RGB-D sequences is:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt CONFIG_FILE DATASET_SEQUENCE_PATH ASSOCIATION_FILE
```

Example for `fr3_sitting_static`:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml datasets/TUM/rgbd_dataset_freiburg3_sitting_static datasets/TUM/rgbd_dataset_freiburg3_sitting_static/associations.txt
```

Example for `fr3_sitting_xyz`:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml datasets/TUM/rgbd_dataset_freiburg3_sitting_xyz datasets/TUM/rgbd_dataset_freiburg3_sitting_xyz/associations.txt
```

Example for `fr3_walking_static`:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml datasets/TUM/rgbd_dataset_freiburg3_walking_static datasets/TUM/rgbd_dataset_freiburg3_walking_static/associations.txt
```

Example for `fr3_walking_rpy`:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml datasets/TUM/rgbd_dataset_freiburg3_walking_rpy datasets/TUM/rgbd_dataset_freiburg3_walking_rpy/associations.txt
```

For other TUM sequences, replace the sequence folder name and YAML configuration file according to the selected dataset.

---

## 10. Running Bonn RGB-D Dynamic Sequences

Please place Bonn RGB-D Dynamic Dataset sequences under:

```text
datasets/Bonn/
```

Example Bonn sequence structure:

```text
datasets/
└── Bonn/
    └── rgbd_bonn_crowd/
        ├── rgb/
        ├── depth/
        ├── groundtruth.txt
        └── associations.txt
```

The Bonn RGB-D Dynamic Dataset follows a structure similar to the TUM RGB-D Dataset. Therefore, it can be run using the RGB-D executable with a Bonn camera configuration file.

The general command format for Bonn RGB-D Dynamic Dataset sequences is:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/Bonn.yaml DATASET_SEQUENCE_PATH ASSOCIATION_FILE
```

Example for `rgbd_bonn_person_tracking`:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/Bonn.yaml datasets/Bonn/rgbd_bonn_person_tracking datasets/Bonn/rgbd_bonn_person_tracking/associations.txt
```

Example for `rgbd_bonn_crowd`:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/Bonn.yaml datasets/Bonn/rgbd_bonn_crowd datasets/Bonn/rgbd_bonn_crowd/associations.txt
```

If `Examples/RGB-D/Bonn.yaml` is not available, please create it according to the camera parameters of the Bonn RGB-D Dynamic Dataset, or modify an existing TUM RGB-D YAML file according to the Bonn dataset calibration.

---

## 11. Running with the Provided Shell Script

An example shell script is provided:

```text
Seqtest
```

Before running the script, please open it and modify the following paths according to your local environment:

```text
Vocabulary path
YAML configuration path
Dataset sequence path
Association file path
YOLO model path if hard-coded in the source code
```

Then run:

```bash
bash Seqtest
```

---

## 12. Notes on Reproducibility

Before running TUM or Bonn sequences, please check the following:

1. `Vocabulary/ORBvoc.txt` has been extracted from `Vocabulary/ORBvoc.zip`.
2. `Thirdparty/libtorch/` has been correctly placed.
3. `best.torchscript` exists in the repository root directory.
4. `voc.names` exists in the repository root directory.
5. The dataset sequence is placed under `datasets/TUM/` or `datasets/Bonn/`.
6. The RGB-D association file exists.
7. The YAML camera configuration file matches the selected dataset.
8. The CUDA version and LibTorch version are compatible.
9. The hard-coded YOLO model path in the source code is consistent with the local repository path, if applicable.

If the YOLO model path is hard-coded in the source file, please modify it according to your local environment before compilation.

---

## 13. Model Information

The provided TorchScript model is used for semantic segmentation-based dynamic-object prior generation. The current setting follows a VOC/SBD 20-class segmentation configuration.

The semantic front-end generates candidate dynamic regions. The final feature filtering is performed by combining semantic information with RGB-D geometric constraints.

---

## 14. Dataset Usage Notes

The public TUM RGB-D and Bonn RGB-D Dynamic Dataset sequences are used for RGB-D SLAM evaluation.

For self-collected RGB-D sequences, please prepare the data in a similar RGB-D format and make sure the RGB and depth frames are correctly synchronized.

The self-collected sequence used in the manuscript is only used for qualitative dense mapping validation and failure-case observation. It is not used for quantitative trajectory evaluation because no external ground-truth trajectory is available.

---

## 15. Important Implementation Notes

This repository is intended for academic research and reproduction.

Some external dependencies and large files are not directly committed to the repository, including:

```text
Thirdparty/libtorch/
Thirdparty/libtorch.rar
Vocabulary/ORBvoc.txt
build/
lib/
datasets/
videos/
```

These files are excluded to keep the repository lightweight and avoid large-file upload limitations.

---
