import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from models import Generator
import random
import os
import time

# --- CONFIGURATION ---
PATIENT_STORIES = [
    "64-year-old male with a history of smoking, presenting with a productive cough.",
    "28-year-old female reports sharp chest pain and persistent chills.",
    "75-year-old patient showing signs of tachypnea.",
    "42-year-old patient with a dry cough and shortness of breath."
]

# --- STEP 1: LOAD THE BRAINS (Generator + Proctor) ---
def load_models():
    try:
        # 1. The Painter (Generator)
        netG = Generator()
        netG.load_state_dict(torch.load('.gen_pneumonia.pth', map_location='cpu'))
        netG.eval()
        
        # 2. The Custom Proctor (Proprietary ResNet18 Classifier)
        # We initialize without weights because we are loading your custom .pth file
        proctor = models.resnet18(weights=None) 
        
        # We must modify the final layer to 2 classes to match your saved state_dict
        num_ftrs = proctor.fc.in_features
        proctor.fc = nn.Linear(num_ftrs, 2)
        
        # Loading your specific trained model
        proctor.load_state_dict(torch.load('vveda_custom_proctor.pth', map_location='cpu'))
        proctor.eval()
        
        print("✅ Proprietary Proctor loaded successfully with 99.7% Accuracy weights.")
        return netG, proctor
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

netG, proctor = load_models()

# --- STEP 2: LOGIC ENGINE ---
def generate_case():
    with torch.no_grad():
        noise = torch.randn(1, 100, 1, 1)
        fake_tensor = netG(noise)
        
        # Real Proctor Audit
        output = proctor(fake_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        
        conf = float(probs.max()) # REAL confidence
        label_idx = torch.argmax(probs).item()
        is_pneumonia = True if label_idx == 1 else False
        
        # Convert tensor to image
        img_array = (fake_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2
        img_array = (img_array * 255).astype(np.uint8)
        
    return img_array, is_pneumonia, conf

def get_heatmap(img_array):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def generate_vveda_proctor_report(diag, confidence):
    conf_pct = confidence * 100
    if diag == "Normal":
        return f"**Proctor Analysis:** Normal scan verified with **{conf_pct:.1f}%** confidence. Lung fields are clear."
    else:
        return f"**Proctor Analysis:** Pneumonia detected with **{conf_pct:.1f}%** confidence. Opacities identified."

import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from models import Generator
import random
import os

# --- CONFIGURATION ---
PATIENT_STORIES = [
    "64-year-old male with a history of smoking, presenting with a productive cough.",
    "28-year-old female reports sharp chest pain and persistent chills.",
    "75-year-old patient showing signs of tachypnea.",
    "42-year-old patient with a dry cough and shortness of breath."
]

# --- STEP 1: LOAD THE BRAINS ---
def load_models():
    try:
        # 1. The Painter (Generator)
        netG = Generator()
        netG.load_state_dict(torch.load('gen_pneumonia.pth', map_location='cpu'))
        netG.eval()
        
        # 2. The Custom Proctor (Using your 99.7% weights)
        proctor = models.resnet18(weights=None)
        num_ftrs = proctor.fc.in_features
        proctor.fc = nn.Linear(num_ftrs, 2)
        proctor.load_state_dict(torch.load('vveda_custom_proctor.pth', map_location='cpu'))
        proctor.eval()
        
        return netG, proctor
    except Exception as e:
        print(f"Deployment Error: {e}")
        return None, None

netG, proctor = load_models()

# --- STEP 2: LOGIC ENGINE ---
def generate_case():
    with torch.no_grad():
        noise = torch.randn(1, 100, 1, 1)
        fake_tensor = netG(noise)
        
        output = proctor(fake_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        
        conf = float(probs.max())
        label_idx = torch.argmax(probs).item()
        # 0 = Normal, 1 = Pneumonia (Based on your training folders)
        is_pneumonia = True if label_idx == 1 else False
        
        img_array = (fake_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2
        img_array = (img_array * 255).astype(np.uint8)
        
    return img_array, is_pneumonia, conf

def get_heatmap(img_array):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- STEP 3: RECONSTRUCTED INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # State tracking
    raw_img_state = gr.State()
    truth_label = gr.State()
    conf_state = gr.State()
    streak_counter = gr.State(value=0)

    gr.Markdown("# 🫁 Vveda: Med-Ed-Gen Proctoring Engine")

    with gr.Row():
        # LEFT SIDE: Clinical Feed
        with gr.Column(scale=2):
            patient_history = gr.Textbox(label="Patient History", interactive=False)
            radiograph_display = gr.Image(label="Live Radiograph", height=450)
            admit_btn = gr.Button("Admit Next Patient", variant="stop")

        # RIGHT SIDE: Diagnostics & Proctor
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### User Diagnosis")
                choice = gr.Radio(["Normal", "Pneumonia"], label="Select Assessment")
                submit_btn = gr.Button("Submit Diagnosis", variant="primary")
            
            with gr.Group():
                gr.Markdown("### Proctor Audit Results")
                proctor_feedback = gr.Markdown("*Awaiting clinical submission...*")
                streak_display = gr.Number(label="Learning Streak", value=0, interactive=False)
                heatmap_display = gr.Image(label="Pathology Heatmap (On Error)", visible=False)

    # --- BUTTON LOGIC ---
    def reset_for_new_patient():
        img, is_p, conf = generate_case()
        history = random.choice(PATIENT_STORIES)
        return {
            radiograph_display: img,
            raw_img_state: img,
            truth_label: is_p,
            conf_state: conf,
            patient_history: history,
            proctor_feedback: "*New patient admitted. Awaiting assessment...*",
            heatmap_display: gr.update(visible=False),
            choice: gr.update(value=None)
        }

    def process_submission(user_choice, is_p, img, streak, conf):
        actual = "Pneumonia" if is_p else "Normal"
        conf_pct = conf * 100
        
        # Determine correctness
        is_correct = (user_choice == actual)
        new_streak = int(streak) + 1 if is_correct else 0
        
        # Create report
        status_icon = "✅ **Correct Assessment!**" if is_correct else "❌ **Incorrect Assessment.**"
        report = (
            f"{status_icon}\n\n"
            f"**Proctor Verdict:** {actual}\n"
            f"**Confidence:** {conf_pct:.1f}%\n"
            f"**Clinical Insight:** Opacities verified by engine." if is_p else 
            f"**Clinical Insight:** Lung fields cleared by engine."
        )
        
        # Show heatmap only on error
        heatmap_visible = gr.update(value=get_heatmap(img) if not is_correct else None, visible=not is_correct)
        
        return report, new_streak, heatmap_visible, new_streak

    admit_btn.click(
        fn=reset_for_new_patient, 
        outputs=[radiograph_display, raw_img_state, truth_label, conf_state, patient_history, proctor_report, heatmap_display, choice]
    )

    submit_btn.click(
        fn=process_submission,
        inputs=[choice, truth_label, raw_img_state, streak_counter, conf_state],
        outputs=[proctor_report, streak_counter, heatmap_display, streak_display]
    )

demo.launch()
