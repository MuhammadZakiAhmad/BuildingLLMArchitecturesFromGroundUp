# Build an LLM (from Scratch)

This repository documents the foundational research and implementation efforts in understanding and constructing Large Language Models (LLMs) from fundamental principles. The primary objective is to bypass high-level abstractions in favor of rigorous, mathematically grounded implementations of state-of-the-art transformer architectures.

## Ongoing Research

### Attention Calibration Distillation for Demonstration-Free Text Reasoning
This directory contains the codebase and empirical findings for my current, active research investigating attention calibration distillation methodologies. The work aims to enhance the inherent text reasoning capabilities of language models in a zero-shot, demonstration-free context.

## Repository Structure

### Core Architectures & Implementations
Ground-up implementations of established LLM architectures to empirically analyze their structural and functional paradigms:
- **`LLM_GPT_2_from_scratch_Sebastiane_Build_an_llm_from_scratch/`**: Implementation of the GPT-2 architecture, building upon Sebastian Raschka's foundational literature.
- **`LLM_gemma_from_scratch_in_python/`**: A native Python implementation of the Gemma model architecture.
- **`LLM_qwen_3_from_scratch_in_pytorch/`**: PyTorch-based construction of the Qwen 3 model architecture.

### Core Mechanisms & Primitives
Isolated implementations of critical transformer sub-components to facilitate deeper mechanistic interpretability and study:
- **`LLM_practice_codes/SELF_ATTENTION_FROM_SCRATCH.PY`**: A foundational implementation of the scaled dot-product Self-Attention mechanism.
- **`LLM_practice_codes/ROPE.PY`**: Implementation of Rotary Positional Embeddings (RoPE), a standard methodology for positional encoding in modern architectures such as LLaMA, Gemma, and Qwen.
- **`LLM_practice_codes/ROPE_MLA.ipynb`**: Jupyter notebook exploring RoPE with Multi-Latent Attention (MLA) variations.

### Vision-Language Models (VLMs)
Explorations into multimodal architectures combining vision and language:
- **`VLM_clip_style_nanovlm/`**: Implementation of a CLIP-style Vision-Language Model.
- **`VLM_paligema_from_scratch/`**: Implementation of a PaLiGEM-style Vision-Language Model from scratch.

## Technical Environment
- **Python**
- **PyTorch**

## Project Objectives
1. Achieve a rigorous, mathematically sound comprehension of contemporary Transformer architectures.
2. Conduct comparative structural analyses between foundational models (GPT-2, Gemma, Qwen).
3. Implement and evaluate modern advancements in attention mechanisms and positional embeddings.
4. Advance ongoing research in attention calibration and demonstration-free reasoning paradigms.

## Computer Vision

### 1. **FootBall Analysis Using Computer Vison**
- **Description:** The FootBall Analysis CV Project utilizes advanced computer vision techniques to analyze football match videos, providing functionalities such as object detection, tracking, team assignment, and ball possession analysis. It leverages YOLO for detection, ByteTrack for tracking and k-means clustering for team assignments, offering a comprehensive tool for sports analytics.
- **Link:** https://github.com/MuhammadZakiAhmad/FootBallAnalysisCVProject.git
- **Technologies:** Python, Tensorflow, OpenCV, YOLOv8, YOLOv5, Supervision, Bytetrack, Pandas, K Mean Clustering

### 2. **EndtoEndChestCancerClassificationUsingMlflowAndDVC**
- **Description:** This project focuses on the development/production aspect of ML project life cycle, in this project we train a DL model to classify Chest Cancer with an image uplaoding user interface.
- **Link:** https://github.com/MuhammadZakiAhmad/EndToEndChestCancerClassificationUsingMlflowAndDVC.git
- **Technologies:** Python, Tensorflow, keras, Mlflow, Dagshub, DVC, AWS, Github Actions

### 3. **AI-Based Patient's Fitness Trainer: Real-time Exercise Monitoring and Repetition Counting**
- **Description:** Combining computer vision and AI, this project creates a real-time patient physiotherapy trainer that uses openCV and the Mediapipe library for pose estimation. The system detects and tracks human poses during exercises, offering real-time feedback and tracking repetitions.
- **Link:** https://github.com/MuhammadZakiAhmad/AITrainer-CV-Application-.git
- **Technologies:** OpenCV, Mediapipe, Python


### 4. **Smart Car Detection and Counting System**
- **Description:** Utilizing YOLO for object detection, this project defines a Region of Interest (ROI) for vehicle counting, employs object tracking techniques, and visualizes vehicle counts in real-time.
- **Link:** https://github.com/MuhammadZakiAhmad/cvProject2_carCounterUsingYOLOv8.git
- **Technologies:** OpenCV, YOLO v8, Python

### 5. **Automatic Car Detection and License Plate Recognition System (ANPR)**
- **Description:** This project aims to develop an intelligent system for detecting vehicles and recognizing their license plates along with reading the number and logging the data using advanced computer vision techniques and machine learning models.
- **Link:** https://github.com/MuhammadZakiAhmad/ANPRUsingYOLOv8.git
- **Technologies:** OpenCV, YOLO v8, EasyOCR, SORT, Python

### 6. **Human Emotion Detection using Deep Learning**
- **Description:** This project involves training an image classifier from scratch on the Kaggle FER-2013 Dataset to detect emotions from facial expressions. The data consists of 48x48 pixel grayscale images of faces, categorized into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Link:** https://github.com/MuhammadZakiAhmad/HumanEmotionDetection.git
- **Technologies:** Python, Jupyter Notebook, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn

### 7. **Digital Mannequin for Online Shops(Virtual Try On)**
- **Description:**  Developed an innovative software that creates a 3D digital model from user-uploaded photos, allowing virtual try-ons of clothing and accessories. The project features dynamic visualization of user models in motion and an extensive, customizable catalog. Enhanced personalization and customer satisfaction by providing accurate body proportion replication. Aimed at reducing return rates and improving online shopping experiences for consumers and retailers..
- **Link:** https://github.com/MuhammadZakiAhmad/DigitalMannequinForOnlineShops.git
- **Technologies:** Python, pytorch3d, pytorch, C#, Unity
