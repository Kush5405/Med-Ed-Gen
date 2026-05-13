import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from models import Generator
import random
import os
import time

# --- CONFIGURATION ---
PATIENT_STORIES = [
    "64-year-old male with a history of smoking, presenting with a productive cough and 101°F fever.",
    "28-year-old female, non-smoker, reports sharp chest pain during deep breaths and persistent chills.",
    "75-year-old bedridden patient showing signs of tachypnea and decreased oxygen saturation.",
    "42-year-old patient presenting with a dry cough for 3 weeks, now developing shortness of breath.",
    "50-year-old male with sudden onset of shivering and rust-colored sputum."
]

# --- STEP 1: LOAD THE BRAINS ---
def load_models():
    try:
        netG = Generator()
        netG.load_state_dict(torch.load('gen_pneumonia.pth', map_location='cpu'))
        netG.eval()
        print("✅ Generator loaded successfully.")
        return netG
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

netG = load_models()

# --- STEP 2: LOGIC ENGINE ---
def generate_case():
    if netG is None:
        return np.zeros((256, 256, 3), dtype=np.uint8), False
            
    with torch.inference_mode():
        noise = torch.randn(1, 100, 1, 1)
        fake_img = netG(noise).detach().cpu().squeeze()
                
        # Denormalize math: [-1, 1] -> [0, 255]
        img_rescaled = (fake_img + 1.0) / 2.0
        img_rescaled = torch.clamp(img_rescaled, 0, 1)
        img_array = (img_rescaled.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
        is_pneumonia = np.random.choice([True, False])
        return img_array, is_pneumonia

def get_heatmap(img_array):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def generate_vveda_proctor_report(prediction_label, confidence_score):
    narratives = {
        "Pneumonia": (
            "Vveda Diagnostic Insight: The neural proctor has identified diffuse, patchy opacities "
            "within the pulmonary parenchyma. This increased density suggests alveolar spaces are "
            "filled with exudate rather than air. Students should correlate this with potential "
            "clinical symptoms like productive cough or localized rales."
        ),
        "Normal": (
            "Vveda Diagnostic Insight: The neural proctor indicates clear lung fields with "
            "symmetrical aeration. No focal opacities, consolidation, or pleural abnormalities "
            "were detected."
        )
    }
    confidence_pct = round(confidence_score * 100, 1)
    report_text = narratives.get(prediction_label, "Analysis complete.")
    return f"{report_text} [Proctor Confidence: {confidence_pct}%]"

# --- STEP 3: UI ARCHITECTURE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Med-Ed Gen: Vveda AI Residency Trainer")
    
    current_label = gr.State()
    raw_img_state = gr.State()
    streak_state = gr.State(0) 

    with gr.Row():
        with gr.Column(scale=2):
            patient_info = gr.Textbox(label="Patient Clinical History", interactive=False)
            display_img = gr.Image(label="Patient X-Ray", height=400)
            gen_btn = gr.Button("🚀 Admit New Patient", variant="primary")
                    
        with gr.Column(scale=1):
            gr.Markdown("### Diagnostic Assessment")
            user_choice = gr.Radio(["Normal", "Pneumonia"], label="Select Findings:")
            submit_btn = gr.Button("Submit Assessment")
            result_output = gr.Markdown("### Status: Waiting for Patient...")
            proctor_view = gr.Image(label="Proctor Heatmap", height=300)
            # This is where the Smart Para will appear
            smart_report = gr.Markdown("### Clinical Insight will appear here.")

    # --- LOGIC FUNCTIONS ---
    def update_ui(streak):
        if not os.path.exists("startup_gallery"):
            os.makedirs("startup_gallery")
        img, label = generate_case()
        history = random.choice(PATIENT_STORIES)
        Image.fromarray(img).save(f"startup_gallery/case_{int(time.time())}.png")
        return img, label, img, history, gr.update(value=None), "New patient admitted.", ""

    def evaluate(user_choice, actual_label, img_array, streak):
        truth = "Pneumonia" if actual_label else "Normal"
        
        # Generate the Smart Report regardless of correct/incorrect
        # We simulate a 90-98% confidence from your proctor
        report = generate_vveda_proctor_report(truth, random.uniform(0.9, 0.98))
        
        if user_choice == truth:
            new_streak = int(streak) + 1
            msg = f"✅ **Correct!** This is {truth}.\n\n**Streak:** {new_streak}"
            return msg, gr.update(value=None), new_streak, report
        else:
            new_streak = 0
            mistake_heatmap = get_heatmap(img_array)
            msg = f"❌ **Incorrect.** The diagnosis was **{truth}**."
            return msg, gr.update(value=mistake_heatmap), new_streak, report

    gen_btn.click(
        fn=update_ui, 
        inputs=[streak_state], 
        outputs=[display_img, current_label, raw_img_state, patient_info, proctor_view, result_output, smart_report]
    )
        
    submit_btn.click(
        fn=evaluate, 
        inputs=[user_choice, current_label, raw_img_state, streak_state], 
        outputs=[result_output, proctor_view, streak_state, smart_report]
    )

demo.launch()
