import open_clip
import cv2
import torch
from PIL import Image
import numpy as np
import os
from torchvision.transforms import functional as F
import tkinter as tk
from tkinter import simpledialog
import threading

# Create model and transforms
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:apple/MobileCLIP-B-OpenCLIP')
tokenizer = open_clip.get_tokenizer('hf-hub:apple/MobileCLIP-B-OpenCLIP')

# Define action classes
actions = [
    "inserting the ink cartridge in the pen body",
    "placing the spring on the ink cartridge",
    "screwing the tip on the pen body"
]
text = tokenizer(actions)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Path to the directory containing extracted frames
frames_dir = "frames"

# Precompute features for extracted frames
frame_features = []
frame_labels = []

for event_dir in os.listdir(frames_dir):
    event_path = os.path.join(frames_dir, event_dir)
    if not os.path.isdir(event_path):
        continue  # Skip non-directory files

    for frame_file in os.listdir(event_path):
        frame_path = os.path.join(event_path, frame_file)
        if not frame_file.lower().endswith(('.jpg', '.png')):
            continue  # Skip non-image files

        # Load and preprocess the frame
        frame_image = Image.open(frame_path).convert("RGB")
        frame_input = preprocess_val(frame_image).unsqueeze(0)

        # Compute features
        with torch.no_grad():
            frame_feature = model.encode_image(frame_input)
            frame_feature /= frame_feature.norm(dim=-1, keepdim=True)

        frame_features.append(frame_feature)
        frame_labels.append(event_dir)

frame_features = torch.cat(frame_features)

# Initialize global variables for average embeddings
average_embeddings = {}
embedding_counts = {}

def add_to_average_embedding(action, frame_embedding):
    global average_embeddings, embedding_counts

    if action not in average_embeddings:
        average_embeddings[action] = frame_embedding
        embedding_counts[action] = 1
    else:
        average_embeddings[action] = (
            average_embeddings[action] * embedding_counts[action] + frame_embedding
        ) / (embedding_counts[action] + 1)
        embedding_counts[action] += 1

def capture_frame_and_update_embedding():
    global model, preprocess_val

    # Get the action from the text box
    action = action_entry.get()
    if not action:
        print("Please enter an action.")
        return

    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return

    # Convert frame to PIL Image and preprocess
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    frame_input = preprocess_val(pil_image).unsqueeze(0)

    # Compute the embedding for the frame
    with torch.no_grad():
        frame_embedding = model.encode_image(frame_input)
        frame_embedding /= frame_embedding.norm(dim=-1, keepdim=True)

    # Update the average embedding for the action
    add_to_average_embedding(action, frame_embedding)
    print(f"Updated average embedding for action: {action}")

# Create the UI
root = tk.Tk()
root.title("Action Embedding UI")

# Create a text box for entering the action
action_label = tk.Label(root, text="Enter Action:")
action_label.pack()
action_entry = tk.Entry(root)
action_entry.pack()

# Create a button to capture the frame and update the embedding
capture_button = tk.Button(root, text="Capture Frame", command=capture_frame_and_update_embedding)
capture_button.pack()

# Start the webcam
cap = cv2.VideoCapture(0)

# Initialize prediction history buffer
prediction_history = []
MAX_HISTORY_LENGTH = 10

def find_best_matching_embedding(image_features):
    """
    Takes an image feature vector and returns the key from average_embeddings 
    that most closely matches the image features, along with the similarity score
    """
    if not average_embeddings:
        return None, 0.0
    
    # Convert average_embeddings dictionary to a tensor
    avg_embeddings_tensor = torch.stack(list(average_embeddings.values()))
    
    # Compute similarity scores
    similarity = (100.0 * image_features @ avg_embeddings_tensor.T.squeeze(1)).softmax(dim=-1)
    
    # Get the best match
    value, index = similarity[0].topk(1)
    best_key = list(average_embeddings.keys())[index.item()]
    confidence = value.item() * 100
    
    return best_key, confidence

def show_webcam():
    global prediction_history
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL Image and preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_input = preprocess_val(pil_image).unsqueeze(0)
    
        # Get predictions
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
    
            # Compute similarity with text features
            similarity_text = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            text_value, text_index = similarity_text[0].topk(1)
            text_prediction = actions[text_index.item()]
            text_confidence = text_value.item() * 100
            
            # Find the best matching average embedding
            embedding_prediction, embedding_confidence = find_best_matching_embedding(image_features)
            
            # Choose between text prediction and average embedding prediction
            if embedding_prediction and embedding_confidence > text_confidence:
                predicted_action = embedding_prediction
                confidence = embedding_confidence
            else:
                predicted_action = text_prediction
                confidence = text_confidence
            
            # Add current prediction to history
            prediction_history.append((predicted_action, confidence))
            # Keep only last MAX_HISTORY_LENGTH predictions
            if len(prediction_history) > MAX_HISTORY_LENGTH:
                prediction_history.pop(0)
            
            # Find most common prediction in history
            if prediction_history:
                from collections import Counter
                most_common_prediction, count = Counter([p[0] for p in prediction_history]).most_common(1)[0]
                avg_confidence = sum([p[1] for p in prediction_history if p[0] == most_common_prediction]) / count
            else:
                most_common_prediction = predicted_action
                avg_confidence = confidence
    
        # Display the most common prediction from history on the frame
        cv2.putText(frame, f"{most_common_prediction}: {avg_confidence:.1f}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Action Recognition', frame)
    
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the webcam feed in a separate thread
webcam_thread = threading.Thread(target=show_webcam)
webcam_thread.daemon = True
webcam_thread.start()

# Run the UI loop
root.mainloop()

# Release the webcam
cap.release()
cv2.destroyAllWindows()
