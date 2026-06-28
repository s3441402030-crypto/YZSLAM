## External Resources and Download Links

### LibTorch

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

Please make sure that the LibTorch version is compatible with the CUDA version used on your system.

---

### TUM RGB-D Dataset

The TUM RGB-D Dataset can be downloaded from the official website:

```text
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
```

The dataset homepage is:

```text
https://cvg.cit.tum.de/data/datasets/rgbd-dataset
```

For RGB-D SLAM evaluation, please download the required RGB-D sequences and their corresponding ground-truth trajectories. An association file is required to synchronize RGB and depth images.

---

### Bonn RGB-D Dynamic Dataset

The Bonn RGB-D Dynamic Dataset can be downloaded from the official website:

```text
https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html
```

This dataset provides dynamic RGB-D sequences for SLAM evaluation in highly dynamic scenes. The sequences follow a format similar to the TUM RGB-D Dataset, so similar evaluation tools can be used.

---

### ORB Vocabulary

The original ORB vocabulary file is provided in compressed form:

```text
Vocabulary/ORBvoc.zip
```

Before building or running the system, extract it:

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
