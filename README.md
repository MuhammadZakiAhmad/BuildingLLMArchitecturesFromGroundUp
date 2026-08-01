# AI Research & Development

Personal and collaborative R&D repository focused on rigorous, mathematically grounded implementations of LLMs, VLMs, and World Models — bypassing high-level abstractions in favor of first-principles understanding.

---

## Active Research

**1. Attention Calibration for Zero-Shot Reasoning via Distillation + LoRA**
Investigating attention calibration distillation to enhance text reasoning in small language models under zero-shot, demonstration-free conditions, using a few-shot teacher model and LoRA fine-tuning.

**2. Thought Transfer Among Multi-Agents via Shared Vector Space**
Human language is inefficient for LLM-to-LLM communication. This project explores vector-level thought transfer between agents with explainability and security as core constraints.

---

## Implementations

### Core Architectures
| Directory | Description |
|---|---|
| `LLM_GPT_2_from_scratch/` | GPT-2 from scratch (Raschka-based) |
| `LLM_gemma_from_scratch/` | Gemma architecture in native Python |
| `LLM_qwen_3_from_scratch/` | Qwen 3 in PyTorch |

### Transformer Primitives
| File | Description |
|---|---|
| `SELF_ATTENTION_FROM_SCRATCH.PY` | Scaled dot-product self-attention |
| `ROPE.PY` | Rotary Positional Embeddings (RoPE) |
| `ROPE_MLA.ipynb` | RoPE + Multi-Latent Attention exploration |

### Vision-Language Models
| Directory | Description |
|---|---|
| `VLM_clip_style_nanovlm/` | CLIP-style VLM |
| `VLM_paligema_from_scratch/` | PaLiGemma-style VLM from scratch |

---

## Computer Vision Projects

### 1. Football Match Analysis · [repo](https://github.com/MuhammadZakiAhmad/FootBallAnalysisCVProject)
Object detection, multi-object tracking, team assignment via K-Means clustering, and real-time ball possession analysis on match video. Uses YOLOv8 for detection and ByteTrack for tracking.
`YOLOv8 · ByteTrack · K-Means · OpenCV · Supervision · Pandas`

### 7. Virtual Try-On (Digital Mannequin) · [repo](https://github.com/MuhammadZakiAhmad/DigitalMannequinForOnlineShops)
Generates a 3D digital model from user-uploaded photos for virtual clothing try-on, with motion visualization and a customizable catalog. Targets reduction of return rates in e-commerce.
`PyTorch3D · PyTorch · Unity · C#`

### 3. AI Physiotherapy Trainer · [repo](https://github.com/MuhammadZakiAhmad/AITrainer-CV-Application-)
Real-time pose estimation for patient exercise monitoring. Detects and tracks human poses to count repetitions and provide live feedback during physiotherapy sessions.
`OpenCV · Mediapipe`

### 4. Smart Car Detection & Counting · [repo](https://github.com/MuhammadZakiAhmad/cvProject2_carCounterUsingYOLOv8)
Defines a Region of Interest for vehicle counting, applies object tracking, and visualizes counts in real-time using YOLOv8.
`YOLOv8 · OpenCV`

### 5. Automatic Number Plate Recognition (ANPR) · [repo](https://github.com/MuhammadZakiAhmad/ANPRUsingYOLOv8)
Detects vehicles, localizes license plates, reads plate numbers via OCR, and logs the data. Combines detection, tracking, and text recognition in a single pipeline.
`YOLOv8 · EasyOCR · SORT · OpenCV`

### 6. Human Emotion Detection · [repo](https://github.com/MuhammadZakiAhmad/HumanEmotionDetection)
Image classifier trained from scratch on FER-2013 (48×48 grayscale faces) to recognize 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
`TensorFlow · Keras · NumPy · Pandas · Matplotlib`

### 7. Chest Cancer Classification (MLflow + DVC) · [repo](https://github.com/MuhammadZakiAhmad/EndToEndChestCancerClassificationUsingMlflowAndDVC)
End-to-end ML pipeline for chest cancer classification from images, with a user-facing upload interface. Covers the full production lifecycle: experiment tracking, data versioning, and cloud deployment.
`TensorFlow · Keras · MLflow · DVC · DagsHub · AWS · GitHub Actions`

---

**Stack:** Python · PyTorch · TensorFlow
