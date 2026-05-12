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
        # Load GAN Generator
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
    # Thresholding to highlight pathology
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- STEP 3: UI ARCHITECTURE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Med-Ed Gen: AI Residency Trainer")
    
    # Global States
    current_label = gr.State()
    raw_img_state = gr.State()
    streak_state = gr.State(0) 

    with gr.Row():
        with gr.Column(scale=2):
            patient_info = gr.Textbox(label="Patient Clinical History", interactive=False)
            display_img = gr.Image(label="Patient X-Ray", height=300)
            gen_btn = gr.Button("🚀 Admit New Patient", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### Diagnostic Assessment")
            user_choice = gr.Radio(["Normal", "Pneumonia"], label="Select Findings:")
            submit_btn = gr.Button("Submit Assessment")
            result_output = gr.Markdown("### Status: Waiting for Patient...")
            proctor_view = gr.Image(label="Proctor Heatmap", height=300)

    # --- LOGIC FUNCTIONS ---
    def update_ui(streak):
        if not os.path.exists("startup_gallery"):
            os.makedirs("startup_gallery")
               
        img, label = generate_case()
        history = random.choice(PATIENT_STORIES)
        
        # Save image for audit
        Image.fromarray(img).save(f"startup_gallery/case_{int(time.time())}.png")
        
        # Return 6 items to match the output list below
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
    gen_btn.click(
        fn=update_ui, 
        inputs=[streak_state], 
        outputs=[display_img, current_label, raw_img_state, patient_info, proctor_view, result_output]
    )
    
    submit_btn.click(
        fn=evaluate, 
        inputs=[user_choice, current_label, raw_img_state, streak_state], 
        outputs=[result_output, proctor_view, streak_state]
    )

demo.launch()
