import torch
import numpy as np
import cv2
from PIL import Image

class TrueGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        # Hook to capture gradients during backward pass
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Hook to capture feature maps during forward pass
        def forward_hook(module, input, output):
            self.features = output

        # Register the hooks on the final convolutional layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx):
        # 1. Forward pass
        output = self.model(input_tensor)
        
        # 2. Target the specific class score
        score = output[0][class_idx]
        
        # 3. Backward pass to compute gradients
        self.model.zero_grad()
        score.backward()

        # 4. Compute channel weights from gradients
        gradients = self.gradients.cpu().data.numpy()[0]
        features = self.features.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))  # Global average pooling

        # 5. Compute weighted combination of features
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i]

        # 6. Apply ReLU (ignore pixels that negatively impact the prediction)
        cam = np.maximum(cam, 0)
        
        # 7. Normalize between 0 and 1
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # Resize to match original image size
        cam = cv2.resize(cam, (224, 224))
        return cam

def apply_heatmap_overlay(original_img_path, cam_heatmap):
    # Read raw image
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))
    
    # Scale heatmap to 0-255 and apply Jet Colormap
    heatmap = np.uint8(255 * cam_heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay the heatmap onto original structural image
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
