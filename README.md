# AI Image Detect App

This project is an Image Detection web app built with Hugging Face Transformers and Gradio. It allows users to upload an image and get predictions for its content using a pretrained vision model.

## Demo

Try the live app on Hugging Face Spaces:  
[https://huggingface.co/spaces/vysubs28/image-detect-app](https://huggingface.co/spaces/vysubs28/image-detect-app)

## Features

- Upload images easily via web UI
- Uses state-of-the-art vision transformer model for accurate classification
- Clean and responsive interface built with Gradio
- Runs on CPU/GPU or Apple Silicon MPS if available

## Installation (optional)

To run locally:

```bash
git clone https://github.com/yourusername/image-detect-app.git
cd image-detect-app
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
python app.py
Usage
Open the app

Upload an image

View the predicted label and confidence score

Technologies Used
Python 3.11

Transformers (Hugging Face)

Gradio

PyTorch

License
MIT License
