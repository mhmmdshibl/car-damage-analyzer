# 🚗 Car Damage Analyzer

An end-to-end AI application for automated vehicle damage detection and damage report generation using a custom-trained YOLOv8 object detection model and OpenAI GPT.

The system detects visible vehicle damage from an uploaded image and converts the detected damage into a concise, human-readable report.

## 🚀 Live Demo

**Hugging Face Spaces:**  
[Add your Hugging Face Space URL here]

**GitHub:**  
https://github.com/mhmmdshibl/car-damage-analyzer

---

## 📌 Project Overview

Vehicle damage assessment is often performed manually by inspecting photographs and identifying damaged areas.

This project implements an automated computer vision pipeline that can:

- Detect visible vehicle damage from an image
- Identify different damage categories
- Provide detection confidence scores
- Generate a natural-language damage report
- Provide basic repair recommendations
- Provide an interactive Gradio web interface

The project combines **computer vision, deep learning, and large language models (LLMs)**.

---

## 🧠 System Architecture

```text
                     Car Image
                         │
                         ▼
                ┌─────────────────┐
                │     YOLOv8      │
                │ Custom CarDD    │
                │     Model       │
                └────────┬────────┘
                         │
                         ▼
                 Damage Detections
                         │
                         ▼
                Detection Processing
                         │
                         ▼
                 Prompt Construction
                         │
                         ▼
                ┌─────────────────┐
                │   OpenAI GPT     │
                │      LLM         │
                └────────┬────────┘
                         │
                         ▼
                Natural-Language
                  Damage Report
                         │
                         ▼
                  Gradio Web App
                         │
                         ▼
                Hugging Face Spaces