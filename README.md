# FCPPE-Net

**FCPPE-Net: Calibrating the Principal Point of Vehicle-Mounted Fisheye Cameras Using Point-Oriented Representation**  
[IEEE SENSORS JOURNAL, 2024]

**GitHub:** https://github.com/BJTU-zjc/FCPPE-Net  

## Overview

FCPPE-Net is a lightweight and accurate deep learning framework for **self-calibrating the principal point** of vehicle-mounted **fisheye cameras** in unconstrained driving scenarios. The model is based on a novel **Point-Oriented Representation**, which encodes geometric priors and enhances feature sensitivity to projection distortions near the principal point.

## Highlights

-  **Point-Oriented Representation**: Projects image features relative to hypothetical principal point centers.
-  **Fisheye-Specific Loss Functions**: Custom loss terms tailored for radial distortion modeling.
-  **Lightweight Architecture**: Real-time capable and hardware-friendly.
-  **Unsupervised & Robust**: No need for manual labels or external calibration targets.

##  Project Structure

FCPPE-Net/
├── datasets/ # Synthetic and real-world fisheye datasets
├── models/ # FCPPE-Net model architecture
└── README.md

## Dataset
Download the dataset from [Baidu Netdisk](https://pan.baidu.com/s/1jFYV5-5n1EqrUb25qwLcrA?pwd=kr4m).
