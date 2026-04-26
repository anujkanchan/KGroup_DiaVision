import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from huggingface_hub import hf_hub_download

# ── Config ──
IMG_SIZE    = 300
DEVICE      = torch.device('cpu')
HF_REPO     = 'Salonideshmukh/dr-detection-model'
MODEL_FILE  = 'best_dr_model.pth'


# ── Custom exception for invalid images ──
class NotFundusImageError(ValueError):
    pass


# ── Fundus image validator ──
def is_fundus_image(img_rgb):
    """
    Checks whether an image looks like a fundus (retinal) photograph.
    Uses 5 heuristics that are true for virtually all fundus images:

    1. Roughly circular/oval bright region on a dark background
       (fundus cameras produce a circular field on black)
    2. The dominant region is reddish-orange (retina colour)
    3. High red-channel dominance in the bright area
    4. Large dark border fraction (fundus cameras always have black padding)
    5. Aspect ratio is close to square (fundus images are always ~1:1)

    Returns (True, "") or (False, reason_string).
    """
    h, w = img_rgb.shape[:2]

    # ── Check 1: aspect ratio must be roughly square (0.5 – 2.0) ──
    aspect = w / h
    if aspect < 0.5 or aspect > 2.0:
        return False, "Image aspect ratio is too extreme for a fundus photograph."

    # Work in BGR for OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Check 2: large dark border — fundus images always have black padding ──
    # At least 5% of pixels must be very dark (border)
    dark_pixels  = np.sum(gray < 20)
    total_pixels = h * w
    dark_fraction = dark_pixels / total_pixels
    if dark_fraction < 0.05:
        return False, "No dark border detected. Fundus images always have a dark circular border."

    # ── Check 3: bright circular region must exist ──
    # Threshold to get the bright "eye" area
    _, bright_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    bright_fraction = np.sum(bright_mask > 0) / total_pixels

    # Fundus: bright area is 30%–90% of the image
    if bright_fraction < 0.30 or bright_fraction > 0.92:
        return False, "The bright region size is inconsistent with a fundus photograph."

    # ── Check 4: bright region must be roughly circular ──
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "No clear retinal disc region found in the image."

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    hull    = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)

    # Solidity — how "filled" the convex shape is (circle = ~1.0)
    solidity = area / (hull_area + 1e-6)
    if solidity < 0.70:
        return False, "The bright region is not circular enough to be a fundus image."

    # Circularity — 4π·area / perimeter²  (circle = 1.0)
    perimeter = cv2.arcLength(largest, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    if circularity < 0.35:
        return False, "The bright region shape does not match a typical fundus photograph."

    # ── Check 5: colour must be reddish-orange in the bright area ──
    # Fundus retina is always dominated by red/orange tones
    bright_region = img_rgb.copy()
    mask_bool = bright_mask > 0

    if mask_bool.sum() == 0:
        return False, "Could not isolate the retinal region for colour analysis."

    r_mean = img_rgb[:, :, 0][mask_bool].mean()
    g_mean = img_rgb[:, :, 1][mask_bool].mean()
    b_mean = img_rgb[:, :, 2][mask_bool].mean()

    # Red channel must dominate — fundus always has R > G > B
    if not (r_mean > g_mean and r_mean > b_mean):
        return False, "Image colour profile does not match a retinal fundus photograph (expected red-dominant tones)."

    # Red channel mean must be reasonably bright (not a black image)
    if r_mean < 30:
        return False, "Image appears too dark to be a valid fundus photograph."

    # Red-to-blue ratio — retina is always warm-toned
    rb_ratio = r_mean / (b_mean + 1e-6)
    if rb_ratio < 1.15:
        return False, "Image colour profile does not match a retinal fundus photograph."

    return True, ""


# ── Model definition (exact same as training) ──
class DRClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0, global_pool=''
        )
        feat_dim = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.gradients  = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        feats = self.backbone(x)
        if feats.requires_grad:
            feats.register_hook(self.save_gradient)
        self.activations = feats
        return self.head(self.pool(feats))


def load_model():
    """Download model from HuggingFace and load weights."""
    import os

    # Step 1: Download weights FIRST (needs network)
    print("  Downloading model weights from HuggingFace Hub...")
    model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)
    print(f"  Weights saved to: {model_path}")

    # Step 2: NOW block timm/transformers from making further outbound calls
    os.environ['TIMM_FUSED_ATTN']      = '0'
    os.environ['HF_DATASETS_OFFLINE']  = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # Step 3: Build model and load weights
    print("  Building model architecture...")
    model = DRClassifier(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("  Model loaded successfully.")
    return model


# ── Preprocessing (identical to training) ──
def crop_black_borders(img, threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y+h, x:x+w]

def ben_graham_normalization(img, sigmaX=10):
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_image(img_array):
    """
    Input  : numpy array in RGB (from PIL / Flask uploader)
    Output : preprocessed numpy array in RGB
    """
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img = crop_black_borders(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = apply_clahe(img)
    img = ben_graham_normalization(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def to_tensor(img_rgb):
    """Normalize and convert to tensor — identical to val_transform in training."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_rgb.astype(np.float32) / 255.0
    img  = (img - mean) / std
    img  = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def predict(model, img_array, threshold=0.35):
    """
    Full inference pipeline.
    img_array : numpy RGB array from PIL/Flask

    Raises NotFundusImageError if the image does not look like a fundus photograph.
    Returns : preprocessed image, grad-cam heatmap, prediction label, probability
    """
    # ── Validate: must be a fundus image before running the model ──
    valid, reason = is_fundus_image(img_array)
    if not valid:
        raise NotFundusImageError(reason)

    # Preprocess
    img_rgb    = preprocess_image(img_array)
    img_tensor = to_tensor(img_rgb)

    # Grad-CAM
    model.eval()
    model.backbone.set_grad_checkpointing(enable=False)

    inp = img_tensor.unsqueeze(0).to(DEVICE)

    features = model.backbone(inp)
    features.retain_grad()
    pooled   = model.pool(features)
    output   = model.head(pooled)

    pred_class = output.argmax(dim=1).item()
    prob       = F.softmax(output, dim=1)[0, 1].item()

    model.zero_grad()
    output[0, pred_class].backward()

    grads   = features.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam     = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam     = cam.squeeze().detach().numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    prediction = 'DR' if prob >= threshold else 'No DR'
    return img_rgb, cam, prediction, prob
