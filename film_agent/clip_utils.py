from utils import print_error

import traceback
import open_clip
import torch
from PIL import Image

# Set device consistently first
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model and move it to the right device
model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:apple/MobileCLIP-B-OpenCLIP')
model = model.to(device)
vision_model = model  # Use the same model instance
print(f"Using device: {device} for CLIP model")
tokenizer = open_clip.get_tokenizer('hf-hub:apple/MobileCLIP-B-OpenCLIP')

def clip_text_encoder(text):
    # Move tokenized text to the same device as the model
    tokenized_text = tokenizer(text).to(device)
    # Process text with model
    with torch.no_grad():
        text_features = model.encode_text(tokenized_text)
    return text_features.squeeze(0).tolist()

def clip_image_encoder(frame):
    # Convert numpy array to PIL Image first
    frame_pil = Image.fromarray(frame)
    
    # Apply MobileCLIP preprocessing
    processed_frame = preprocess_val(frame_pil).unsqueeze(0).to(device)
    
    # Extract CLIP embeddings
    with torch.no_grad():
        try:
            image_features = vision_model.encode_image(processed_frame)
        except RuntimeError as e:
            print_error(f"Model inference failed: {e}")
            traceback.print_exc()
    return image_features.squeeze(0).tolist()