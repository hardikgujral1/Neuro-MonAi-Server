# 🧠 NeuroMONAI Server
> **AI-Powered Brain Tumor Segmentation using MONAI Label**

NeuroMONAI Server is a **deep learning–based medical image segmentation web server** built on **[MONAI Label](https://monai.io/)**.  
It allows **automated and interactive segmentation of brain tumors** from **CT or MRI** scans using MONAI Zoo / Bundle models.

---

## 🚀 Features

- 🔬 **Automatic Brain Tumor Segmentation** (supports T1, T2, T1c, FLAIR, and CT modalities)
- 🧩 **Integration with 3D Slicer** for visualization and editing
- ⚙️ **Pretrained MONAI Bundles** support (use models from [MONAI Model Zoo](https://monai.io/model-zoo))
- 🔁 **Batch inference and active learning** support
- 💻 **Simple local or remote web deployment**

---
---

## 🧩 Prerequisites

Before you begin, ensure you have:

- **Python ≥ 3.10.3**
- **PyTorch ≥ 1.10** (with CUDA if using GPU)
- **MONAI Label ≥ 1.5.0**
- **3D Slicer (optional)** with MONAI Label Plugin

> 💡 Tip: Use a virtual environment (`venv` or `conda`) to keep dependencies isolated.
