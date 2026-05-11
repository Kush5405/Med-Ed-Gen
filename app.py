import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from models import Generator # Your existing models.py
import random
import os
import time


# A library of clinical stories to add "WebVeda" level depth
PATIENT_STORIES = [
    "64-year-old male with a history of smoking, presenting with a productive cough and 101°F fever.",
    "28-year-old female, non-smoker, reports sharp chest pain during deep breaths and persistent chills.",
    "75-year-old bedridden patient showing signs of tachypnea and decreased oxygen saturation.",
    "42-year-old patient presenting with a dry cough for 3 weeks, now developing shortness of breath.",
    "50-year-old male with sudden onset of shivering and rust-colored sputum."
]

# --- STEP 1: LOAD THE BRAINS ---
def load_models():
    # 1. Load GAN Generator
    netG = Generator()
    netG.load_state_dict(torch.load('gen_pneumonia.pth', map_location='cpu'))
    netG.eval()
    
    # 2. Load CNN Detector (The Proctor)
    # Ensure you define your CNN architecture or import it here
    # detector = YourCNNArchitecture()
    # detector.load_state_dict(torch.load('detector.pth', map_location='cpu'))
    # detector.eval()
    
    return netG #, detector (uncomment when detector.pth is ready)

netG = load_models()

# --- STEP 2: LOGIC ENGINE ---
def generate_case():
    with torch.inference_mode():
        noise = torch.randn(1, 100, 1, 1)
        fake_img = netG(noise).detach().cpu().squeeze()
        
        # Denormalize - Keep it at native 256x256
        img_array = ((fake_img.permute(1, 2, 0).numpy() + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        is_pneumonia = np.random.choice([True, False])
        return np.array(img), is_pneumonia
    
def get_heatmap(img_array):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Increase this number! 
    # If 150 was too sensitive, try 190 or 210.
    # This forces the AI to only highlight the most intense 'white' patches.
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- STEP 3: UI ARCHITECTURE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Med-Ed Gen: AI Residency Trainer")
    
    # --- GLOBAL STATES ---
    current_label = gr.State()
    raw_img_state = gr.State()
    streak_state = gr.State(0) 

    with gr.Row():
        with gr.Column(scale=2):
            patient_info = gr.Textbox(label="Patient Clinical History", interactive=False, placeholder="Generate a case to see history...")
            display_img = gr.Image(label="Patient X-Ray", width=256)
            gen_btn = gr.Button("🚀 Admit New Patient", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### Diagnostic Assessment")
            user_choice = gr.Radio(["Normal", "Pneumonia"], label="Select Findings:")
            submit_btn = gr.Button("Submit Assessment")
            
            result_output = gr.Markdown("### Status: Waiting for Patient...")
            proctor_view = gr.Image(label="Proctor Heatmap", visible=True, height=256, width=256)

    # --- LOGIC FUNCTIONS ---
    def update_ui(current_streak):
        if not os.path.exists("startup_gallery"):
            os.makedirs("startup_gallery")
               
        img, label = generate_case()
        history = random.choice(PATIENT_STORIES)
        
        # Save image to gallery
        Image.fromarray(img).save(f"startup_gallery/case_{int(time.time())}.png")
        
        # RETURN 6 ITEMS (Matches the outputs list below)
        return img, label, img, history, gr.update(value=None), "New patient admitted. Review history and X-ray."

    def evaluate(user_choice, actual_label, img_array, streak):
        truth = "Pneumonia" if actual_label else "Normal"
        
        if user_choice == truth:
            new_streak = int(streak) + 1
            level = (new_streak // 5) + 1
            
            # Set titles based on streak
            if new_streak >= 20: rank = "👑 Chief Radiologist"
            elif new_streak >= 10: rank = "🥈 Senior Registrar"
            elif new_streak >= 5: rank = "🥉 Junior Resident"
            else: rank = "👨‍⚕️ Medical Student"

            msg = f"✅ **Correct!** This is {truth}.\n\n"
            msg += f"**Rank:** {rank}\n"
            msg += f"**Level:** {level} | **Streak:** {new_streak}"
            
            # We return gr.update(value=None) because there is no heatmap to show!
            return msg, gr.update(value=None), new_streak
            
        else:
            new_streak = 0
            # We only generate the heatmap image here
            mistake_heatmap = get_heatmap(img_array)
            msg = f"❌ **Incorrect.** The diagnosis was **{truth}**.\n\nStreak reset to 0."
            
            return msg, gr.update(value=mistake_heatmap), new_streak
            
    # --- BUTTON CONNECTIONS ---
    
    # 1 input (streak), 6 outputs
    gen_btn.click(fn=update_ui, inputs=[streak_state], outputs=[display_img, current_label, raw_img_state, patient_info, proctor_view, result_output])
    
    # 4 inputs, 3 outputs
    submit_btn.click(fn=evaluate, inputs=[user_choice, current_label, raw_img_state, streak_state], outputs=[result_output, proctor_view, streak_state])

demo.launch()
