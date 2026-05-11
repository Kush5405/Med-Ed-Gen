🏥 Med-Ed Gen: AI Radiology Residency Trainer 🩺


(Above: Real-time demonstration of the Med-Ed Gen interface synthesizing synthetic pulmonary pathology and providing XAI proctoring.)

An end-to-end generative AI platform leveraging Deep Convolutional GANs (DCGAN) to synthesize privacy-safe, high-fidelity chest X-rays for clinical diagnostic training and residency simulation.

🎯 Project Overview:

Med-Ed Gen addresses the critical bottleneck in medical education: the scarcity of diverse, accessible, and privacy-compliant clinical datasets. By utilizing adversarial training, this system provides an infinite stream of synthetic X-ray cases (Normal vs. Pneumonia), paired with dynamic clinical histories.

The platform achieves a high degree of anatomical realism and includes a proprietary "AI Proctor" layer that utilizes Explainable AI (XAI) to provide students with immediate spatial feedback on their diagnostic errors.

Note on Intellectual Property: This repository serves as a technical portfolio demonstration for VIT Bhopal University. To protect pending intellectual property and the proprietary mathematical weights developed during the research phase, the specific .pth model weights and raw clinical training datasets are not publicly distributed.

✨ Key Features:

Generative Pathology Synthesis: Custom-trained DCGAN architecture capable of generating unique 256×256 pulmonary scans on demand.

Explainable AI (XAI) Proctor: Integrated heatmap visualization system that localizes "density clusters" to teach students where pathology exists.

Clinical Story Engine: A dynamic randomization pipeline that pairs scans with symptomatic histories (e.g., fever, smoking history, sputum types).

Gamified Residency Loop: A full-stack training simulation featuring Milestone Badges (Junior Resident → Chief Radiologist) and Infinite Leveling.

Interactive Web Deployment: A low-latency Gradio interface optimized for cross-device clinical simulation.

Live Case Gallery: Automated local storage system (startup_gallery) for archiving unique synthetic cases for further study.

🏗️ System Architecture:

Phase 1: Generative Adversarial Pipeline:
Generator Network: A 5-layer transposed convolutional architecture that transforms a 100-D latent noise vector into structured 256x256 grayscale X-rays.
Discriminator Network: A deep CNN trained to differentiate between real pulmonary scans and synthetic outputs, driving the Generator toward anatomical perfection.

Phase 2: Pedagogical Logic Layer:
The Clinical Router: Assigns "Normal" or "Pneumonia" labels to generated frames and pairs them with high-fidelity patient clinical histories.

The Feedback Loop: If a student misdiagnoses, the system triggers the Proctor Heatmap using pixel-intensity thresholding to reveal the underlying pathology.

Phase 3: Deployment & UX:
Frontend: Gradio 4.0+ themed with "Medical Soft" styling.

Backend: PyTorch-driven inference engine optimized for consumer-grade hardware .

📁 Repository Structure:




🔬 Technical Specifications:

Custom DCGAN Parameters:
  Input: 100-dimensional latent vector (z)
  Optimizer: Adam (\beta_1=0.5, \beta_2=0.999)
  Activation: Leaky ReLU (Discriminator), ReLU (Generator)
  Normalization: Batch Normalization across all convolutional layers
  Output Resolution: 256×256×1 Grayscale

XAI Proctor Logic:
  Method: Spatial Density 
  LocalizationVisualization: JET Color Map 
  OverlaySensitivity: Pixel Intensity Thresholding (T \ge 200)
  Logic: Triggered exclusively upon incorrect diagnostic submission to facilitate active learning.

🎓 Academic Context:

Institution: VIT Bhopal University

Student: Kushagra Singh

Major: B.Tech Computer Science & AI (2023-2027)

Research Goals:

   Evaluate GAN utility in reducing medical data scarcity.

   Develop "Human-in-the-Loop" AI training interfaces.

   Explore synthetic image fidelity in pulmonary diagnostics.

📈 Performance & Milestones:

Metric :                                                        Achievement :
Synthesis Resolution                                           256x256 Native
Inference Latency                                              < 0.5s (CPU)
Gamification Depth                                             Level 1 - ∞
Deployment                                                     Local & Hugging Face Spaces


🛑 Copyright & Licensing
© 2026 Kushagra Singh. All Rights Reserved.

This repository is provided strictly for academic portfolio demonstration. The proprietary code, architectural concepts, and generative logic herein may not be copied, reproduced, distributed, or utilized for commercial purposes without explicit written permission from the author.

Patent Pending / IP Protected.

⚠️ Academic Use Only | Not for Clinical Diagnosis | Research Demonstration Only




