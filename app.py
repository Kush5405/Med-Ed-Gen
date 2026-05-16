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
from datetime import datetime
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
PATIENT_STORIES = [
    "64-year-old male with a history of smoking, presenting with a productive cough.",
    "28-year-old female reports sharp chest pain and persistent chills.",
    "75-year-old patient showing signs of tachypnea.",
    "42-year-old patient with a dry cough and shortness of breath."
]

# --- STEP 1: LOAD MODELS ---
def load_models():
    try:
        from huggingface_hub import hf_hub_download

        # Securely fetching paths from your private model repository
        gen_path = hf_hub_download(repo_id="Kush5405/vveda-weights-vault", filename="gen_pneumonia.pth")
        proctor_path = hf_hub_download(repo_id="Kush5405/vveda-weights-vault", filename="vveda_custom_proctor.pth")

        # Loading Generator from the secure local cache path
        netG = Generator()
        netG.load_state_dict(torch.load(gen_path, map_location='cpu'))
        netG.eval()
        
        # Loading Proctor from the secure local cache path
        proctor = models.resnet18(weights=None) 
        num_ftrs = proctor.fc.in_features
        proctor.fc = nn.Linear(num_ftrs, 2)
        proctor.load_state_dict(torch.load(proctor_path, map_location='cpu'))
        proctor.eval()
        
        return netG, proctor
    except Exception as e:
        print(f"Error loading models from secure repository: {e}")
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

# --- REVISED INTERFACE (Step 3) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    raw_img_state = gr.State()
    truth_label = gr.State()
    conf_state = gr.State()
    streak_counter = gr.State(value=0)

    gr.Markdown("# Vveda: Med-Ed-Gen Proctoring Engine")

    with gr.Row():
        with gr.Column(scale=2):
            patient_info = gr.Textbox(label="Patient History", interactive=False)
            display_img = gr.Image(label="Live Radiograph", height=400)
            admit_btn = gr.Button("Admit Next Patient", variant="stop")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Clinical Assessment")
                choice = gr.Radio(["Normal", "Pneumonia"], label="Select Diagnosis")
                submit_btn = gr.Button("Submit Diagnosis", variant="primary")
            
            with gr.Group():
                proctor_analysis = gr.Markdown("### Proctor Insight")
                streak_display = gr.Number(label="Learning Streak", value=0, interactive=False)
                
                # CHANGE: visible is now TRUE by default so the space exists
                heatmap_display = gr.Image(label="Visual Pathology Map", visible=True, height=250)
                
                gr.Markdown("---")
                gr.Markdown("### 📥 Archive Clinical Records")
                xray_download = gr.File(label="Raw X-Ray (.png)")
                heatmap_download = gr.File(label="Heatmap (.png)")

    def update_for_new_patient():
        img, is_p, conf = generate_case()
        history = random.choice(PATIENT_STORIES)
        return {
            display_img: img,
            raw_img_state: img,
            truth_label: is_p,
            conf_state: conf,
            patient_info: history,
            proctor_analysis: "### Proctor Insight\n*Patient admitted. Awaiting Assessment...*",
            choice: gr.update(value=None),
            # Reset the heatmap to None so the space looks empty for the new patient
            heatmap_display: None, 
            xray_download: None,
            heatmap_download: None
        }

    def evaluate_diagnosis(user_choice, is_p, img, streak, conf):
        actual = "Pneumonia" if is_p else "Normal"
        conf_pct = conf * 100
        correct = (user_choice == actual)
        new_streak = int(streak) + 1 if correct else 0
        
        status = "✅ **Correct Assessment!**" if correct else "❌ **Incorrect Assessment.**"
        report = f"### {status}\n**Proctor Verdict:** {actual} ({conf_pct:.1f}% confidence)."
        
        heatmap_img = get_heatmap(img)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xray_name = f"Xray_{ts}_{actual}.png"
        map_name = f"Heatmap_{ts}_{actual}.png"
        
        Image.fromarray(img).save(xray_name)
        Image.fromarray(heatmap_img).save(map_name)
        
        return {
            proctor_analysis: report,
            streak_counter: new_streak,
            streak_display: new_streak,
            # Simply update the value; visibility is already True
            heatmap_display: heatmap_img,
            xray_download: xray_name,
            heatmap_download: map_name
        }

    # Connect buttons (No changes needed here)
    admit_btn.click(fn=update_for_new_patient, outputs=[display_img, raw_img_state, truth_label, conf_state, patient_info, proctor_analysis, choice, heatmap_display, xray_download, heatmap_download])
    submit_btn.click(fn=evaluate_diagnosis, inputs=[choice, truth_label, raw_img_state, streak_counter, conf_state], outputs=[proctor_analysis, streak_counter, streak_display, heatmap_display, xray_download, heatmap_download])

# Import the verify function from your new auth file
from auth import authenticate, AUTH_MSG

# Update the launch command
demo.launch(
    auth=authenticate, 
    auth_message=AUTH_MSG
)
