# Banana Ripeness Classification: Deep Learning and Transfer Learning

## Overview

This repository contains an end-to-end computer vision pipeline developed to classify agricultural produce—specifically bananas—into distinct ripeness categories (Ripe and Unripe). 

The project was created to evaluate, benchmark, and deploy automated image classification models. It contrasts a lightweight Custom Convolutional Neural Network (CNN) built entirely from scratch against an industry-standard pre-trained architecture (MobileNetV3) utilizing both feature extraction and fine-tuning methodologies. Additionally, the repository includes a fully functional, production-ready Streamlit web application for real-time model inference.

---

## Technical Architecture and Methodology

The pipeline addresses image classification through the following core stages:

1. **Data Pipeline and Preprocessing:** Automated dataset ingestion, resizing to standard spatial dimensions, and internal data augmentation/preprocessing pipelines.
2. **Custom CNN Architecture:** A compact, parameter-efficient convolutional neural network engineered from scratch to optimize performance under strict memory constraints.
3. **Transfer Learning (MobileNetV3):** Leveraging a pre-trained MobileNetV3 backbone as a fixed feature extractor by freezing base layers and training a custom classification head.
4. **Fine-Tuning:** Unfreezing the top 30 layers of the MobileNetV3 backbone and continuing training at a reduced learning rate (10x smaller) to adapt pre-trained features to the target domain.
5. **Deployment:** A Streamlit interface designed for local execution and remote hosting.

---

## Experimental Results

All models were evaluated on an identical, unseen test dataset consisting of 264 images. 

| Model Architecture | Test Accuracy | Total Parameters | Key Characteristic |
| :--- | :---: | :---: | :--- |
| **Custom CNN** | 100.00% | 322,338 | High efficiency and optimal parameter footprint |
| **MobileNetV3 (Feature Extraction)** | 100.00% | 1,087,346 | Stable baseline leveraging pre-trained representations |
| **MobileNetV3 (Fine-Tuned)** | 100.00% | 1,087,346 | Domain-adapted feature refinement |

---

## Repository Structure

```text
mobilenetv3_transfer.keras
app.py                  # Streamlit web application
requirements.txt        # Pinned dependency versions
README.md
