# Drone Detection Optimization (AU-AIR Dataset)

This repository contains the optimized implementation for high-fidelity object detection in aerial (drone) imagery using the AU-AIR dataset. The core novelty of this project is the transition from heavy legacy architectures to a lightweight, high-resolution **YOLOv26m Turbo-HD** protocol.

## 🚀 Model Performance Comparison

| Feature | YOLOv3-SPP (Previous) | YOLOv26m (Baseline) | **YOLOv26m Turbo-HD (Boosted)** |
| :--- | :--- | :--- | :--- |
| **Model Size** | 62.61 Million | 21.78 Million | **21.78 Million** |
| **GFLOPs** | 117.2 | 74.8 | **191.4** |
| **Input Res** | 640 x 640 | 640 x 640 | **1024 x 1024** |
| **mAP50 Accuracy**| 35.9% | 36.3% | **38.7% ** |
| **Efficiency** | Legacy Baseline | Balanced Performance | **High Fidelity Detection** |
| **Optimization** | None | Altitude Normalization| **Alt-Aware + 1024px** |

## 🛠️ Key Improvements & Novelty

### 1. Architectural Efficiency
We significantly reduced the model complexity by transitioning to **YOLOv26m**. This yielded a **65% reduction in parameters** (21.78M vs 62.61M) while simultaneously improving the baseline detection accuracy.

### 2. Turbo-HD Protocol (1024px High-Res)
Standard models often struggle with extremely small objects in drone footage. We implemented a **Turbo-HD** push:
*   **1024px Resolution**: Capturing fine-grained details of tiny objects.
*   **Rectangular Training**: Optimized for the wide aspect ratio of AU-AIR drone cameras, resulting in a **1.7x speedup** in training efficiency.

### 3. Altitude-Aware Performance
The model leverages altitude-normalized training data, ensuring consistent performance regardless of the drone's height.

### 4. Unified Detection Strategy
By applying strategic weighting and high-resolution optimizations across all 8 classes, we pushed the "Unified" mAP50 to a peak of **38.7%**.

## 💻 Core Scripts
*   `turbo_hd_1024.py`: Primary optimization and high-resolution training logic.
*   `validate_master.py`: Specialized validation script for master accuracy metrics.
*   `models.py`: Architecture definitions for the YOLOv26m series.

---
*Developed for Advanced Drone Traffic Monitoring Efficiency.*
