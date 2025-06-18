import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import gradio as gr
from PIL import Image

# Set device (MPS for Apple Silicon, CUDA if on other GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load better model: Swin Transformer
model_name = "microsoft/swin-base-patch4-window7-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

# Prediction function
def classify_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)[0]

    # Top 5 predictions
    top5 = torch.topk(probs, 5)
    top_labels = [model.config.id2label[idx.item()] for idx in top5.indices]
    top_scores = [round(conf.item(), 4) for conf in top5.values]

    return dict(zip(top_labels, top_scores))  # format for gr.Label

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AI Image Classifier
        Upload any image and get accurate predictions using a state-of-the-art model by Microsoft.
        """
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image", show_label=True)
        label_output = gr.Label(num_top_classes=5, label="Predictions")

    submit_btn = gr.Button("Classify Image", variant="primary")
    submit_btn.click(fn=classify_image, inputs=image_input, outputs=label_output)

# Launch the app
demo.launch(share=True)



