#region Imports
import sys
from pathlib import Path
import time
import os
import glob

sys.path.append(str(Path(__file__).parents[1]))

import random
import gymnasium as gym
from types import SimpleNamespace
import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

# Import LIDA components
from lidapy.utils import Node
from lidapy.agent import Environment 
from lidapy.acs import AttentionCodelet
from lidapy.csm import CurrentSituationalModel
from lidapy.pam import PerceptualAssociativeMemory
from lidapy.global_workspace import GlobalWorkspace
from lidapy.ss import SensoryMemory, SensorySystem
from lidapy.ps import ProceduralMemory, ProceduralSystem, SchemeUnit
from lidapy.sms import MotorPlan, SensoryMotorMemory, SensoryMotorSystem

from film_agent.pam import DefaultPAMMemory
#endregion
#region Environment
class FilmEnvironment(Environment):
    """Environment that captures video from camera and records based on movement detection.
    
    Uses CLIP embeddings to compare current frames with reference frames of 
    throwing/not-throwing to determine when to start and stop recording.
    """
    def __init__(self, render_mode=None, video_source=0, 
                 throwing_reference_folder=None, not_throwing_reference_folder=None,
                 output_dir="recordings"):
        self.action_space = gym.spaces.Discrete(2)  # Two actions: record/stop
        self.is_recording = False
        self.cap = cv2.VideoCapture(video_source)
        self.reference_frame = None
        self.current_frame = None
        self.output_dir = output_dir
        self.video_writer = None
        self.current_video_path = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load reference frames for comparing throwing/not-throwing actions
        self.throwing_references = []
        self.not_throwing_references = []
        
        # Load all throwing reference images from folder
        if throwing_reference_folder and os.path.isdir(throwing_reference_folder):
            self.throwing_references = self.load_images_from_folder(throwing_reference_folder)
            print(f"Loaded {len(self.throwing_references)} throwing reference images from {throwing_reference_folder}")
                    
        # Load all not-throwing reference images from folder
        if not_throwing_reference_folder and os.path.isdir(not_throwing_reference_folder):
            self.not_throwing_references = self.load_images_from_folder(not_throwing_reference_folder)
            print(f"Loaded {len(self.not_throwing_references)} not-throwing reference images from {not_throwing_reference_folder}")

    def load_images_from_folder(self, folder_path):
        """Load all images from a folder"""
        images = []
        # Look for common image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            # Also look in subdirectories
            image_paths.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
        
        for image_path in image_paths:
            img = self.load_reference_image(image_path)
            if img is not None:
                images.append(img)
                
        return images

    def load_reference_image(self, image_path):
        """Load a reference image from a file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image from {image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading reference image {image_path}: {e}")
            return None

    def start_recording(self):
        """Initialize VideoWriter and start recording"""
        if self.is_recording:
            return  # Already recording
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.current_video_path = os.path.join(self.output_dir, f"recording_{timestamp}.avi")
        
        # Get dimensions of the frame
        height, width = self.current_frame.shape[:2]
        

        fps = 30.0
        
        self.video_writer = cv2.VideoWriter(
            self.current_video_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps,
            (width, height)
        )
                
        if not self.video_writer.isOpened():
            print(f"Error: Could not create video writer for {self.current_video_path}")
            self.video_writer = None
            return
        
        self.is_recording = True
        print(f"Started recording to {self.current_video_path}")
        
        # Write the current frame (first frame)
        if self.current_frame is not None:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_frame)

    def record_frame(self):
        """Add current frame to the video"""
        if not self.is_recording or self.video_writer is None or self.current_frame is None:
            return
        
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(bgr_frame)

    def stop_recording(self):
        """Stop recording and release VideoWriter"""
        if not self.is_recording or self.video_writer is None:
            return
        
        self.is_recording = False
        self.video_writer.release()
        print(f"Stopped recording. Video saved to {self.current_video_path}")
        self.video_writer = None

    def run_commands(self, commands):
        """Execute motor commands for recording control"""
        if commands is None:
            return
        
        if 'record' in commands:
            if commands['record'] == 0:  # Start recording
                self.start_recording()
            else:  # Stop recording
                self.stop_recording()
                
        return self.is_recording
    
    def receive_sensory_stimuli(self):
        """Read frame from video feed and convert to RGB format"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # If we're recording, add this frame to the video
        if self.is_recording and self.video_writer is not None:
            self.record_frame()
            
        return self.current_frame

    def close(self):
        """Clean up resources"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        if self.cap is not None:
            self.cap.release()
            
        self.is_recording = False
        return None
#endregion


# Load pre-trained CLIP model for frame comparison
model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:apple/MobileCLIP-B-OpenCLIP')
# Move model to CPU to avoid CUDA memory issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = model.to(device)
print(f"Using device: {device} for CLIP model")

# Cache for reference embeddings and average category embeddings
reference_embeddings = {}
category_embeddings = {
    "throwing_avg": None,
    "not_throwing_avg": None
}

def compute_average_embedding(embeddings_list):
    """Compute the average embedding from a list of embeddings"""
    if not embeddings_list:
        return None
        
    # Stack all embeddings and compute the mean
    stacked = torch.cat([emb[0].features for emb in embeddings_list], dim=0)
    avg_embedding = torch.mean(stacked, dim=0, keepdim=True)
    # Normalize the average embedding
    avg_embedding /= avg_embedding.norm(dim=-1, keepdim=True)
    
    # Create a node to hold the average embedding
    avg_node = Node(content="average_features", activation=1.0)
    avg_node.features = avg_embedding
    
    return [avg_node]

def vision_processor(frame, identifier=None):
    """Process a video frame using CLIP to extract semantic features
    
    Args:
        frame: The frame to process
        identifier: Optional string identifier for caching embeddings
    """
    global reference_embeddings
    
    if frame is None:
        return []
    
    # Check cache for reference frames
    if identifier and identifier in reference_embeddings:
        return reference_embeddings[identifier]
    
    try:
        # Convert numpy array to PIL Image first
        frame_pil = Image.fromarray(frame)
        
        # Apply MobileCLIP preprocessing
        processed_frame = preprocess_val(frame_pil).unsqueeze(0)
        
        # Move to same device as the model
        processed_frame = processed_frame.to(device)
        
        # Extract CLIP embeddings
        with torch.no_grad():
            try:
                image_features = vision_model.encode_image(processed_frame)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            except RuntimeError as e:
                print(f"Error during model inference: {e}")
                print("Trying with smaller batch...")
                # If we hit memory issues, try processing with CPU
                cpu_model = vision_model.to('cpu')
                cpu_frame = processed_frame.to('cpu')
                image_features = cpu_model.encode_image(cpu_frame)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # Move back to original device
                vision_model.to(device)
                image_features = image_features.to(device)
        
        # Create Node with features for comparison
        node = Node(content="frame_features", activation=1.0)
        node.features = image_features
        result = [node]
        
        # Cache result if identifier was provided
        if identifier:
            reference_embeddings[identifier] = result
            
        return result
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return []

# Define similarity function for nodes
def similarity_function(node1, node2):
    return node1.similarity(node2)

Node.similarity_function = classmethod(similarity_function)

# Set up LIDA components
sensors = [{"name": "vision_sensor", "modality": "image", "processor": vision_processor}]
actuators = [{"name": "record", "modality": "video", "processor": None}]
pam = PerceptualAssociativeMemory(memory=DefaultPAMMemory())
sm = SensoryMemory(sensors=sensors)

def action_function(dorsal_update=None):
    """Convert perception nodes to motor commands for recording actions"""
    if dorsal_update:
        for node in dorsal_update:
            if node.content == "throwing":
                return {"record": 0}  # Start recording
            elif node.content == "not throwing":
                return {"record": 1}  # Stop recording
    
    return {"record": None}

# Create motor plans and schemes
mps = [
    MotorPlan("record", action_function)
]
contexts = [None, 'G', 'H']
schemes = [SchemeUnit(context=[Node(content=context, activation=1)] if context else None, action=mp) 
           for context, mp in zip(contexts, mps)] 

# Set up cognitive components
acs = [AttentionCodelet()]
pm = ProceduralMemory(schemes=schemes)

# Assemble LIDA agent
lida_agent = {
    'sensory_system': SensorySystem(pam=pam, sensory_memory=sm),
    'csm': CurrentSituationalModel(),
    'gw': GlobalWorkspace(attention_codelets=acs, broadcast_receivers=[pm]),
    'procedural_system': ProceduralSystem(procedural_memory=pm),
    'sensory_motor_system': SensoryMotorSystem(actuators=actuators, motor_plans=mps),
}

def minimally_conscious_agent(environment, lida_agent, steps=100, similarity_threshold=0.8):
    """Run the cognitive cycle of the LIDA agent for a specified number of steps"""
    lida_agent = SimpleNamespace(**lida_agent)
    current_motor_commands = {'record': 0}
    
    # Process all reference images and compute category averages
    throwing_embeddings = []
    not_throwing_embeddings = []
    
    # Process throwing references
    print(f"Processing {len(environment.throwing_references)} throwing reference images...")
    for i, ref_img in enumerate(environment.throwing_references):
        emb = vision_processor(ref_img, identifier=f"throwing_ref_{i}")
        if emb:
            throwing_embeddings.append(emb)
    
    # Process not-throwing references
    print(f"Processing {len(environment.not_throwing_references)} not-throwing reference images...")
    for i, ref_img in enumerate(environment.not_throwing_references):
        emb = vision_processor(ref_img, identifier=f"not_throwing_ref_{i}")
        if emb:
            not_throwing_embeddings.append(emb)
    
    # Compute average embeddings for each category
    if throwing_embeddings:
        category_embeddings["throwing_avg"] = compute_average_embedding(throwing_embeddings)
        print("Computed average throwing embedding")
    
    if not_throwing_embeddings:
        category_embeddings["not_throwing_avg"] = compute_average_embedding(not_throwing_embeddings)
        print("Computed average not-throwing embedding")
    
    # Main cognitive cycle
    for _ in range(steps):
        # Perception phase
        current_stimuli = environment.receive_sensory_stimuli()
        if current_stimuli is None:
            break
            
        # Process current frame and create percepts
        associated_nodes = lida_agent.sensory_system.process(current_stimuli)
        
        # CLIP-based frame comparison
        current_features = vision_processor(current_stimuli)
        
        if current_features:
            throwing_similarity = 0
            not_throwing_similarity = 0
            
            # Compare with throwing category average
            if category_embeddings["throwing_avg"]:
                throwing_tensor = category_embeddings["throwing_avg"][0].features
                current_tensor = current_features[0].features
                throwing_similarity = torch.nn.functional.cosine_similarity(
                    throwing_tensor, current_tensor
                ).item()
            
            # Compare with not-throwing category average
            if category_embeddings["not_throwing_avg"]:
                not_throwing_tensor = category_embeddings["not_throwing_avg"][0].features
                current_tensor = current_features[0].features
                not_throwing_similarity = torch.nn.functional.cosine_similarity(
                    not_throwing_tensor, current_tensor
                ).item()
                    
            # Create nodes based on similarity scores
            throwing_node = Node(content=f"throwing_similarity_{throwing_similarity:.2f}", 
                               activation=throwing_similarity)
            not_throwing_node = Node(content=f"not_throwing_similarity_{not_throwing_similarity:.2f}", 
                                   activation=not_throwing_similarity)
            associated_nodes.append(throwing_node)
            associated_nodes.append(not_throwing_node)

            # Make decision based on which category has higher similarity
            decision = "throwing" if throwing_similarity > not_throwing_similarity else "not throwing"
            decision_node = Node(content=decision, activation=1.0)
            associated_nodes.append(decision_node)
            
            # Print debug info for the first few frames
            if _ < 5:
                print(f"Frame {_}: Throwing: {throwing_similarity:.4f}, Not-throwing: {not_throwing_similarity:.4f}, Decision: {decision}")

        # Update current situational model with new percepts
        lida_agent.csm.run(associated_nodes)
        
        # Broadcast to global workspace
        winning_coalition = lida_agent.gw.run(lida_agent.csm)

        # Action selection
        if winning_coalition is not None:
            selected_behavior = lida_agent.procedural_system.run(winning_coalition)
            if selected_behavior is not None:
                # Execute selected actions
                current_motor_commands = lida_agent.sensory_motor_system.run(
                    selected_behavior=selected_behavior, 
                    dorsal_update=associated_nodes,
                    winning_coalition=winning_coalition
                )
                current_motor_commands = lida_agent.sensory_motor_system.get_motor_commands()
                
                # Execute recording commands
                environment.run_commands(current_motor_commands)

if __name__ == '__main__':
    # Initialize environment with reference image folders
    env = FilmEnvironment(
        video_source=0,
        throwing_reference_folder=r"C:\Users\nmdig\CVReaserch\Vector-LIDA\film_agent\frames\throwing",
        not_throwing_reference_folder=r"C:\Users\nmdig\CVReaserch\Vector-LIDA\film_agent\frames\not_throwing",
        output_dir=r"C:\Users\nmdig\CVReaserch\Vector-LIDA\film_agent\recordings"
    )
    try:
        minimally_conscious_agent(env, lida_agent, steps=100)
    finally:
        env.close()