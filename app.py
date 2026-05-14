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

# --- STEP 1: LOAD THE BRAINS (Generator + Proprietary Proctor) ---
def load_models():
    try:
        netG = Generator()
        netG.load_state_dict(torch.load('gen_pneumonia.pth', map_location='cpu'))
        netG.eval()
        
        proctor = models.resnet18(weights=None) 
        num_ftrs = proctor.fc.in_features
        proctor.fc = nn.Linear(num_ftrs, 2)
        proctor.load_state_dict(torch.load('vveda_custom_proctor.pth', map_location='cpu'))
        proctor.eval()
        
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

# --- STEP 3: RECONSTRUCTED INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    raw_img_state = gr.State()
    truth_label = gr.State()
    conf_state = gr.State()
    streak_counter = gr.State(value=0)

    gr.Markdown("# 🫁 Vveda: Med-Ed-Gen Proctoring Engine")

    with gr.Row():
        # LEFT COLUMN
        with gr.Column(scale=2):
            patient_info = gr.Textbox(label="Patient History", interactive=False)
            display_img = gr.Image(label="Live Radiograph", height=450)
            admit_btn = gr.Button("Admit Next Patient", variant="stop")

        # RIGHT COLUMN
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Clinical Assessment")
                choice = gr.Radio(["Normal", "Pneumonia"], label="Select Diagnosis")
                submit_btn = gr.Button("Submit Diagnosis", variant="primary")
            
            with gr.Group():
                proctor_analysis = gr.Markdown("### Proctor Insight\n*Awaiting submission...*")
                streak_display = gr.Number(label="Learning Streak", value=0, interactive=False)
                
                heatmap_display = gr.Image(
                    label="Pathology Heatmap", 
                    visible=False, 
                    container=False, 
                    interactive=False,
                    show_label=True,
                    height=300
                )

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
            # FIX: We reset the value to None but don't force visible=False here
            heatmap_display: gr.update(value=None) 
        }

    def evaluate_diagnosis(user_choice, is_p, img, streak, conf):
        actual = "Pneumonia" if is_p else "Normal"
        conf_pct = conf * 100
        correct = (user_choice == actual)
        new_streak = int(streak) + 1 if correct else 0
        
        status = "✅ **Correct Assessment!**" if correct else "❌ **Incorrect Assessment.**"
        report = f"### {status}\n**Proctor Verdict:** {actual} detected ({conf_pct:.1f}% confidence)."
        
        heatmap_img = get_heatmap(img)
        
        return {
            proctor_analysis: report,
            streak_counter: new_streak,
            streak_display: new_streak,
            heatmap_display: gr.update(value=heatmap_img, visible=True)
        }

    # FIX: These were previously indented inside the function, preventing them from triggering
    admit_btn.click(
        fn=update_for_new_patient, 
        outputs=[display_img, raw_img_state, truth_label, conf_state, patient_info, proctor_analysis, choice, heatmap_display]
    )

    submit_btn.click(
        fn=evaluate_diagnosis,
        inputs=[choice, truth_label, raw_img_state, streak_counter, conf_state],
        outputs=[proctor_analysis, streak_counter, streak_display, heatmap_display]
    )

demo.launch()
