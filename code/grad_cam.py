import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import types 
from PIL import Image

def patch_densenet_forward(model):
    def new_forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False) 
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    model.forward = types.MethodType(new_forward, model)

def generate_gradcam(model, input_tensor, img_path, output_dir):
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    # Apply Patch
    if "DenseNet" in model.__class__.__name__:
        patch_densenet_forward(model)

    model.eval()
    for param in model.parameters(): param.requires_grad = True 
    
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach() 

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    target_layer = None
    if hasattr(model, 'features'): 
        target_layer = getattr(model.features, 'norm5', model.features[-1])
    elif hasattr(model, 'backbone'):
        target_layer = model.backbone.layer4[-1]
    
    if target_layer is None: return np.zeros((224, 224))
        
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        model.zero_grad()
        score.backward()

        if activations is None or gradients is None: return np.zeros((224, 224))

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        if cam.max() > 0: cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        # --- SAVE IMAGES (Original + Heatmap + Overlay) ---
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(img)
        
        heatmap_color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        
        # 1. Save Original
        cv2.imwrite(os.path.join(abs_output_dir, "original.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        # 2. Save Heatmap
        cv2.imwrite(os.path.join(abs_output_dir, "heatmap.png"), heatmap_color)
        # 3. Save Overlay
        cv2.imwrite(os.path.join(abs_output_dir, "overlay.png"), overlay)
        
        print(f"  [SUCCESS] Saved original, heatmap, and overlay to {abs_output_dir}")

    except Exception as e:
        print(f"    [ERROR] Grad-CAM failed: {e}")
        cam = np.zeros((224, 224))

    finally:
        handle_f.remove()
        handle_b.remove()

    return cam