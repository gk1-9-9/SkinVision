# SkinVision: Skin Condition Analysis using YOLOv8, InceptionV3, and Qwen2-VLM

## Overview

**SkinVision** is a comprehensive solution for analyzing skin conditions, specifically focusing on acne detection and jaundice prediction. The project leverages advanced machine learning models—**YOLOv8** for acne detection and **InceptionV3** for jaundice prediction. Additionally, it integrates **Qwen2-VLM** to provide personalized advice based on the analysis results. The application is deployed using the Flask web framework, ensuring an accessible and user-friendly interface.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Project Structure

The repository is organized as follows:
```
SkinVision/
├── models/
├── src/
│   ├── templates
│   ├── __init__.py
│   ├── forms.py
│   ├── image_processing.py
│   ├── llm.py
│   ├── models.py
│   └── routes.py
├── requirements.txt
├── run.py
└── dlib-19.24.1-cp311-cp311-win_amd64.whl
```
## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```
   git clone https://github.com/CarnageOP10/SkinVision-Acne-Jaundice-Prediction-with-YOLOv8-InceptionV3-and-Qwen2-VLM.git
   cd SkinVision-Acne-Jaundice-Prediction-with-YOLOv8-InceptionV3-and-Qwen2-VLM
   ```
   
2. **Create a virtual environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```
4. **Install the dependencies**:
   ```
   pip install -r "req.txt"
   ```
## Usage

get your roboflow and huggingface api keys
```
python run.py
```
