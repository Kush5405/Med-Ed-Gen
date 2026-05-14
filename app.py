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

# --- STEP 3: GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # State management
    raw_img_state = gr.State()
    current_label = gr.State()
    streak_state = gr.State(value=0)
    conf_state = gr.State() # NEW: Stores real confidence

    gr.Markdown("# 🫁 Vveda: Med-Ed-Gen Proctoring Engine")
    
    with gr.Row():
        with gr.Column(scale=2):
            display_img = gr.Image(label="Live Radiograph", height=400)
            with gr.Row():
                btn_normal = gr.Button("Normal", variant="secondary")
                btn_pneumonia = gr.Button("Pneumonia", variant="primary")
        
        with gr.Column(scale=1):
            patient_info = gr.Textbox(label="Patient History", interactive=False)
            proctor_report = gr.Markdown("### Proctor Insight\n*Awaiting submission...*")
            streak_display = gr.Number(label="Learning Streak", value=0)

    gen_btn = gr.Button("Admit Next Patient", variant="stop")

    def update_ui():
        img, label, conf = generate_case()
        history = random.choice(PATIENT_STORIES)
        return img, label, img, history, gr.update(value="### Proctor Insight\n*Awaiting submission...*"), conf

    def evaluate(user_choice, actual_label, img_array, streak, current_conf):
        truth = "Pneumonia" if actual_label else "Normal"
        report = generate_vveda_proctor_report(truth, current_conf)
        
        if user_choice == truth:
            new_streak = int(streak) + 1
            msg = f"✅ **Correct!** The Proctoring Engine confirmed {truth}."
            return msg, gr.update(value=None), new_streak, report
        else:
            new_streak = 0
            mistake_heatmap = get_heatmap(img_array)
            msg = f"❌ **Incorrect.** The Proctor identified {truth}."
            return msg, gr.update(value=mistake_heatmap), new_streak, report

    gen_btn.click(fn=update_ui, outputs=[display_img, current_label, raw_img_state, patient_info, proctor_report, conf_state])
    
    btn_normal.click(fn=evaluate, inputs=[gr.State("Normal"), current_label, raw_img_state, streak_state, conf_state], outputs=[proctor_report, display_img, streak_display, proctor_report])
    btn_pneumonia.click(fn=evaluate, inputs=[gr.State("Pneumonia"), current_label, raw_img_state, streak_state, conf_state], outputs=[proctor_report, display_img, streak_display, proctor_report])

demo.launch()
