from utils import print_error

import traceback
import open_clip
import torch
from PIL import Image

model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:apple/MobileCLIP-B-OpenCLIP')
# Move model to CPU to avoid CUDA memory issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = model.to(device)
print(f"Using device: {device} for CLIP model")
tokenizer = open_clip.get_tokenizer('hf-hub:apple/MobileCLIP-B-OpenCLIP')

def clip_text_encoder(text):
    text_features = model.encode_text(tokenizer(text)).squeeze(0)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def clip_image_encoder(frame):
    # Convert numpy array to PIL Image first
    frame_pil = Image.fromarray(frame)
    
    # Apply MobileCLIP preprocessing
    processed_frame = preprocess_val(frame_pil).unsqueeze(0).to(device)
    
    # Extract CLIP embeddings
    with torch.no_grad():
        try:
            image_features = vision_model.encode_image(processed_frame)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        except RuntimeError as e:
            print_error(f"Model inference failed: {e}")
            traceback.print_exc()
    return image_features