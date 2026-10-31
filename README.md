# MASt3R-CNN-ORBSLAM3-1030

Pure-visual research implementation combining MASt3R dense geometry, a lightweight dynamic-risk CNN, learned sparse observations, and native ORB-SLAM3 map and optimization components.

See [Chinese V2 documentation](README_纯视觉V2.md) and [the detailed framework](figures/04_visual_v2_complete.svg).

Includes source, build scripts, controlled geometry/relocalization/loop tests and experimental LoRA + CNN joint-training code. Datasets, pretrained weights, training checkpoints, caches and compiled outputs are excluded. Dependency revisions and upstream repositories are recorded in SOURCES.json; upstream license files are retained in third_party.

The dynamic model and training recipe are experimental. Successful execution does not establish improved SLAM accuracy. The supplied dynamic training sample has only two independent scenes and does not provide a separate labeled dynamic test scene. PnP/RANSAC and the C++ BA backend are not differentiated through during joint training.

## Setup

The tested environment is WSL Ubuntu 22.04 with CUDA-enabled PyTorch, OpenCV, Eigen and Pangolin. Local interpreter and dataset paths in the scripts must be adjusted for other machines. Run bootstrap.py to restore pinned upstream sources, configure_native.py to apply native integration, and build.sh to compile. Obtain the official MASt3R checkpoint separately as documented by the upstream repository. See the Chinese documentation for launch commands and actual validation boundaries.
