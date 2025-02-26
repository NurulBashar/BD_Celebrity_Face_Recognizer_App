import sys
import os
from fastai.vision.all import *
import gradio as gr
import torch
import platform
import pathlib

# Fix Windows path issues
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Debugging info
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"OS: {platform.system()}")

# Celebrity labels
CELEBRITY_LABELS = [
    'Afran Nisho only Face', 'Afsana Mimi', 'Arfin Shuvoo only Face', 'Ayub Bachchu only Face',
    'Chanchal Chowdhury', 'Dr. Muhammad Yunus', 'Fazle Hasan Abed only Face', 'Humayun Ahmed',
    'James (Nagar Baul)', 'Joya Ahsan only Face', 'Mashrafe Bin Mortaza', 'Mizanur Rahman Azhari only Face',
    'Mostofa Sarwar Farooki', 'Sabina Khatun', 'Sabina Yasmin only Face', 'Shakib Al Hasan', 'Shakib Khan',
    'Tahsan Khan', 'Tamim Iqbal'
]

def load_model(model_path):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_learner(model_path, cpu=True)

try:
    # Define model path
    MODEL_PATH = os.path.join(os.getcwd(), "celebrity_model-v44.pkl")
    
    print(f"Loading model from: {MODEL_PATH}")
    learn = load_model(MODEL_PATH)
    print("Model loaded successfully!")

    def predict_celebrity(img):
        """Process image and return predictions."""
        try:
            img = PILImage.create(img)
            _, _, probs = learn.predict(img)
            return {CELEBRITY_LABELS[i]: float(probs[i]) for i in range(len(CELEBRITY_LABELS))}
        except Exception as e:
            return {"error": str(e)}

    # Gradio UI
    iface = gr.Interface(
        fn=predict_celebrity,
        inputs=gr.Image(label="Upload Face Image", type="pil"),
        outputs=gr.Label(label="Recognition Results", num_top_classes=5),
        examples=[
            "jara.jpg", "nisah.jpg", "sabina_yesmin.jpeg",
            "sakib_c.jpg", "tamim.jpg"
        ],
        title="BD Celebrity Face Recognition",
        description="AI-powered Bangladeshi Celebrity Identification System"
    )

    print("Launching application...")
    iface.launch(server_name="0.0.0.0", share=True)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
