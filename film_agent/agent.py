#region Imports
import sys
from pathlib import Path
import time
import os
import glob
import threading
import queue
from collections import deque
import traceback
import platform

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

# Add color formatting for error messages
def print_error(message):
    """Print error message in red color"""
    # ANSI color codes for red text
    RED = '\033[91m'
    RESET = '\033[0m'
    
    # Enable ANSI color codes on Windows
    if platform.system() == 'Windows':
        os.system('')  # Enables ANSI escape sequences in Windows terminal
    
    # Print the error message in red
    print(f"{RED}ERROR: {message}{RESET}")

# Add custom exceptions for better error handling
class RecordingError(Exception):
    """Exception raised for errors in the recording process."""
    pass

class ThreadError(Exception):
    """Exception raised for thread-related errors."""
    pass

class FrameProcessingError(Exception):
    """Exception raised for errors in frame processing."""
    pass

#region Environment
class FilmEnvironment(Environment):
    def __init__(self, video_source=0, 
                 throwing_reference_folder=None, not_throwing_reference_folder=None,
                 output_dir="recordings", fps=30.0):
        self.action_space = gym.spaces.Discrete(2)
        self.is_recording = False
        self.cap = cv2.VideoCapture(video_source)
        self.current_frame = None
        self.output_dir = output_dir
        self.video_writer = None
        self.current_video_path = None
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0

        self.recording_thread = None
        self.recording_queue = queue.Queue(maxsize=60)
        self.should_record = False
        self.thread_active = False
        self.thread_lock = threading.Lock()
        self.last_heartbeat_time = 0

        os.makedirs(output_dir, exist_ok=True)

        self.throwing_references = self.load_images_from_folder(throwing_reference_folder)
        self.not_throwing_references = self.load_images_from_folder(not_throwing_reference_folder)

    def load_images_from_folder(self, folder_path):
        if not folder_path or not os.path.isdir(folder_path):
            return []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = [p for ext in image_extensions for p in glob.glob(os.path.join(folder_path, '**', ext), recursive=True)]
        return [self.load_reference_image(p) for p in image_paths if self.load_reference_image(p) is not None]

    def load_reference_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
        except Exception:
            return None

    def record_frame(self):
        if not self.is_recording or self.current_frame is None:
            return
        now = time.time()
        if now - self.last_frame_time >= self.frame_interval:
            try:
                self.recording_queue.put_nowait(self.current_frame.copy())
                self.last_frame_time = now
            except queue.Full:
                pass

    def stop_recording(self):
        with self.thread_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self.should_record = False

        self.recording_queue.put(None)

        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5)
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved to {self.current_video_path}")
            self.video_writer = None

    def start_recording(self):
        with self.thread_lock:
            if self.is_recording:
                return
            if self.recording_thread and self.recording_thread.is_alive():
                self.thread_active = False
                self.recording_thread.join()

            if self.current_frame is None:
                raise RecordingError("No current frame available to start recording")

            h, w = self.current_frame.shape[:2]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.current_video_path = os.path.join(self.output_dir, f"recording_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(self.current_video_path, fourcc, self.fps, (w, h))

            if not self.video_writer.isOpened():
                raise RecordingError("Failed to open video writer")

            self.should_record = True
            self.thread_active = True
            self.is_recording = True
            self.last_frame_time = time.time()

            with self.recording_queue.mutex:
                self.recording_queue.queue.clear()

            self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
            self.recording_thread.start()

    def _recording_worker(self):
        while self.thread_active:
            with self.thread_lock:
                self.last_heartbeat_time = time.time()
            try:
                frame = self.recording_queue.get(timeout=0.05)
                if frame is None:
                    self.recording_queue.task_done()
                    break
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(bgr_frame)
                self.recording_queue.task_done()
            except queue.Empty:
                time.sleep(0.005)

        if self.video_writer:
            self.video_writer.release()
        self.video_writer = None
        self.thread_active = False

    def close(self):
        if self.is_recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()

    # Explicitly implement required abstract methods
    def receive_sensory_stimuli(self):
        """Read frame with rate-limiting for more consistent processing"""
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise FrameProcessingError("Failed to read frame from camera")
            
            # Convert to RGB format
            try:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise FrameProcessingError(f"Error converting frame: {e}")
            
            # If we're recording, add this frame to video queue
            if self.is_recording and self.thread_active:
                try:
                    self.record_frame()
                except Exception as e:
                    raise RecordingError(f"Error recording frame: {e}")
            
            return self.current_frame # TODO attach to a sensor
        except Exception as e:
            print_error(f"{e}")
            traceback.print_exc()
            return None

    def run_commands(self, commands):
        """Execute motor commands for recording control
        
        This method is required by the Environment abstract base class.
        
        Args:
            commands: Dictionary containing commands
            
        Returns:
            Current recording state
        """
        if commands is None:
            return self.is_recording

        if 'record' in commands:
            if commands['record'] == 0:  # Start recording
                self.start_recording()
            elif commands['record'] == 1:  # Stop recording
                self.stop_recording()
            # If None, maintain current state
    
        return self.is_recording
        
    def step(self, action):
        """Take a step in the environment based on the action
        
        This method is required by some Environment implementations.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Execute the action (record or not)
        if action == 0:  # Record
            self.start_recording()
        elif action == 1:  # Stop recording
            self.stop_recording()
            
        # Get next observation
        observation = self.receive_sensory_stimuli()
        reward = 0  # No reward in this environment
        done = observation is None  # Done if no more frames
        info = {"is_recording": self.is_recording}
        
        return observation, reward, done, info
        
    def reset(self):
        """Reset the environment
        
        This method is required by some Environment implementations.
        
        Returns:
            Initial observation
        """
        # Stop any active recording
        if self.is_recording:
            self.stop_recording()
            
        # Clear any cached state
        self.current_frame = None
        
        # Get first observation
        return self.receive_sensory_stimuli()
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
                print_error(f"Model inference failed: {e}")
                traceback.print_exc()
                raise FrameProcessingError(f"CLIP model inference error: {e}")
        
        # Create Node with features for comparison
        node = Node(content="frame_features", activation=1.0)
        node.features = image_features
        result = [node]
        
        # Cache result if identifier was provided
        if identifier:
            reference_embeddings[identifier] = result
            
        return result
        
    except Exception as e:
        print_error(f"Frame processing failed: {e}")
        traceback.print_exc()
        raise FrameProcessingError(f"Vision processing error: {e}")

# Define direct similarity function for nodes
def direct_cosine_similarity(node1, node2):
    """Calculate cosine similarity between two nodes directly using their features"""
    if hasattr(node1, 'features') and hasattr(node2, 'features'):
        try:
            similarity = torch.nn.functional.cosine_similarity(
                node1.features, node2.features
            ).item()
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    return 0.0

# Replace the recursive similarity function with a direct implementation
def similarity_function(cls, node1, node2):
    """Class method for node similarity calculation that avoids recursion
    
    Args:
        cls: The class (automatically provided when used as a class method)
        node1: First Node object
        node2: Second Node object
    """
    return direct_cosine_similarity(node1, node2)

Node.similarity_function = classmethod(similarity_function)

# Set up LIDA components
sensors = [{"name": "vision_sensor", "modality": "image", "processor": vision_processor}]
actuators = [{"name": "record", "modality": "video", "processor": None}]
pam = PerceptualAssociativeMemory(memory=DefaultPAMMemory())
sm = SensoryMemory(sensors=sensors)

def action_function(dorsal_update=None):
    """Convert perception nodes to motor commands based on content"""
    if dorsal_update:
        for node in dorsal_update:
            # Direct check for content
            if isinstance(node.content, str):
                if "throwing" == node.content:
                    print("ACTION FUNCTION: Detected throwing - starting recording")
                    return {"record": 0}  # Start recording
                elif "not throwing" == node.content:
                    print("ACTION FUNCTION: Detected not throwing - stopping recording")
                    return {"record": 1}  # Stop recording
    
    return {"record": None}  # No change in recording state

# Replace the individual action functions with a single action function that directly checks node content
mps = [MotorPlan("record", action_function)]

# Create a single scheme with no context to be more direct
schemes = [SchemeUnit(context=None, action=mps[0])]

pm = ProceduralMemory(schemes=schemes)
acs = [AttentionCodelet()]

lida_agent = {
    'sensory_system': SensorySystem(pam=pam, sensory_memory=sm),
    'csm': CurrentSituationalModel(),
    'gw': GlobalWorkspace(attention_codelets=acs, broadcast_receivers=[pm]),
    'procedural_system': ProceduralSystem(procedural_memory=pm),
    'sensory_motor_system': SensoryMotorSystem(actuators=actuators, motor_plans=mps),
}

def minimally_conscious_agent(environment, lida_agent, steps=100):
    """Run the cognitive cycle of the LIDA agent for a specified number of steps"""
    lida_agent = SimpleNamespace(**lida_agent)
    
    # Process all reference images and compute category averages
    throwing_embeddings = []
    not_throwing_embeddings = []
    
    # Process throwing references
    print(f"Processing {len(environment.throwing_references)} throwing reference images...")
    for i, ref_img in enumerate(environment.throwing_references):
        try:
            emb = vision_processor(ref_img, identifier=f"throwing_ref_{i}")
            if emb:
                throwing_embeddings.append(emb)
        except Exception as e:
            print_error(f"Failed to process reference image {i}: {e}")
            # Continue with other reference images
    
    # Process not-throwing references
    print(f"Processing {len(environment.not_throwing_references)} not_throwing reference images...")
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
            if throwing_similarity > not_throwing_similarity:
                decision = "throwing"
            else:
                decision = "not throwing"
                
            # Use a high activation to ensure it's considered important
            decision_node = Node(content=decision, activation=0.99)
            associated_nodes.append(decision_node)
            
            print(f"Frame {_}: Throwing: {throwing_similarity:.4f}, Not-throwing: {not_throwing_similarity:.4f}, Decision: '{decision}'")
            print(f"Action based on decision: {'Start Recording' if decision == 'throwing' else 'Stop Recording'}")

        # Update current situational model with new percepts
        lida_agent.csm.run(associated_nodes)
        
        # Broadcast to global workspace
        winning_coalition = lida_agent.gw.run(lida_agent.csm)

        # Action selection 
        if winning_coalition is not None:
            # Fix: Check if winning_coalition is a Coalition object with nodes attribute
            if hasattr(winning_coalition, 'nodes'):
                print(f"Winning coalition nodes: {[node.content for node in winning_coalition.nodes if hasattr(node, 'content')]}")
            else:
                print(f"Winning coalition: {winning_coalition}")
            
            # Run action directly based on winning coalition decisions
            motor_commands = {"record": None}  # Default: no change
            
            if hasattr(winning_coalition, 'nodes'):
                for node in winning_coalition.nodes:
                    if hasattr(node, 'content'):
                        if node.content == "throwing":
                            print(f"Directly starting recording based on 'throwing' decision")
                            motor_commands = {"record": 0}
                            break
                        elif node.content == "not throwing":
                            print(f"Directly stopping recording based on 'not throwing' decision")
                            motor_commands = {"record": 1}
                            break
            
            # Keep the regular behavior system as a backup
            selected_behavior = lida_agent.procedural_system.run(winning_coalition)
            if selected_behavior is not None:
                try:
                    scheme_context = "None" if selected_behavior.scheme.context is None else [
                        n.content for n in selected_behavior.scheme.context if hasattr(n, 'content')
                    ]
                    print(f"Selected behavior: context={scheme_context}, action={selected_behavior.scheme.action_stream[0].action.name}")
                except Exception as e:
                    print(f"Error displaying scheme context: {e}")
                    print(f"Selected behavior: {selected_behavior}")
                
                behavior_commands = lida_agent.sensory_motor_system.run(
                    selected_behavior=selected_behavior, 
                    dorsal_update=associated_nodes,
                    winning_coalition=winning_coalition
                )
                behavior_commands = lida_agent.sensory_motor_system.get_motor_commands()
                
                # Only use behavior commands if they're not None
                if behavior_commands and 'record' in behavior_commands and behavior_commands['record'] is not None:
                    motor_commands = behavior_commands
            
            # Execute the determined commands
            print(f"Final motor commands: {motor_commands}")
            environment.run_commands(motor_commands)

if __name__ == '__main__':
    # Initialize environment with reference image folders and reduced FPS for better playback
    try:
        env = FilmEnvironment(
            video_source=0,
            throwing_reference_folder=r"C:\Users\nmdig\CVReaserch\Vector-LIDA\film_agent\frames\throwing",
            not_throwing_reference_folder=r"C:\Users\nmdig\CVReaserch\Vector-LIDA\film_agent\frames\not_throwing",
            output_dir=r"C:\Users\nmdig\CVReaserch\Vector-LIDA\film_agent\recordings",
            fps=30
        )
        
        try:
            minimally_conscious_agent(env, lida_agent, steps=1000)
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...")
        except Exception as e:
            print_error(f"Agent execution failed: {e}")
            traceback.print_exc()
        finally:
            print("Cleaning up resources...")
            # Force immediate cleanup without waiting for queue
            with env.thread_lock:
                env.thread_active = False
                env.should_record = False
                env.is_recording = False
                
                # Force clear any pending queue items
                with env.recording_queue.mutex:
                    env.recording_queue.queue.clear()
                    
            # Now close properly
            env.close()
            print("Done!")
    except Exception as e:
        print_error(f"Environment initialization failed: {e}")
        traceback.print_exc()