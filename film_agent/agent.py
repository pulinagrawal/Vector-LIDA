#region Imports
import traceback
import torch
import sys
import numpy as np
import cv2
from pathlib import Path
from torch import tensor
from PIL import Image

sys.path.append(str(Path(__file__).parents[1]))


# Import LIDA components
from lidapy.utils import Node
from lidapy.agent import minimally_conscious_agent
from lidapy.acs import AttentionCodelet
from lidapy.csm import CurrentSituationalModel
from lidapy.pam import PerceptualAssociativeMemory
from lidapy.global_workspace import GlobalWorkspace
from lidapy.ss import SensoryMemory, SensorySystem
from lidapy.ps import ProceduralMemory, ProceduralSystem, SchemeUnit
from lidapy.sms import MotorPlan, SensoryMotorMemory, SensoryMotorSystem

from film_agent.env import FilmEnvironment
from film_agent.pam import DefaultPAMMemory

from film_agent.utils import compute_average_embedding, print_error
from film_agent.utils import similarity_function 
from film_agent.clip_utils import clip_image_encoder, clip_text_encoder

def vision_processor(frame):
    """Process a video frame using CLIP to extract semantic features
    
    Args:
        frame: The frame to process
    """
    if frame is None:
        return []

    image_features = clip_image_encoder(frame)
    
    # Create Node with features for comparison
    node = Node(content="frame features", activation=1.0)
    node.features = image_features
    result = [node]
    
    return result
        
# This is just a wrapper that forwards to the function in utils.py
def compute_average_embedding(embeddings_list, ema_mode=True, prev_embedding=None, env=None):
    """Compute the average embedding from a list of embeddings.
    
    This is a wrapper for the main implementation in utils.py.
    
    Args:
        embeddings_list: List of embeddings to average
        ema_mode: If True, use exponential moving average instead of simple average
        prev_embedding: Previous EMA value (only used if ema_mode=True)
        env: FilmEnvironment instance to get EMA parameters from
    
    Returns:
        The averaged embedding (or EMA updated embedding if in EMA mode)
    """
    # Forward to the implementation in utils.py
    from film_agent.utils import compute_average_embedding as compute_avg_emb_impl
    return compute_avg_emb_impl(embeddings_list, ema_mode, prev_embedding, env)

def combine_features(self, node1, node2) -> list:
    """Combine features of two nodes by averaging them"""
    combined_features = compute_average_embedding([node1, node2])
    return combined_features

Node.similarity_function = classmethod(similarity_function)
Node.combine_features_function = classmethod(combine_features)


# Set up LIDA components
sensors = [{"name": "vision_sensor", "modality": "image"}]
feature_detectors = [vision_processor]
actuators = [{"name": "record", "modality": "video", "processor": None}]
pam = PerceptualAssociativeMemory(memory=DefaultPAMMemory())
sm = SensoryMemory(sensors=sensors, feature_detectors=feature_detectors)

def record_function(dorsal_update=None):
    """Convert perception nodes to motor commands based on content"""
    return {"record": 0, "display": "recording"}  # start recording

def stop_function(dorsal_update=None):
    """Convert perception nodes to motor commands based on content"""
    return {"record": 1, "display": "not recording"}  # stop recording
    
# Replace the individual action functions with a single action function that directly checks node content
mps = [MotorPlan("record", record_function), 
       MotorPlan("stop", stop_function)
      ]  

def action_node(content):
    node = Node(content=content, activation=1.0)
    node.features = clip_text_encoder(content)
    return node

def frame_node(frame):
    node = Node(content="frame features", activation=1.0)
    node.features = clip_image_encoder(frame)
    return node

action1_node = action_node("throwing")
action2_node = action_node("not throwing")
frame1_result = frame_node(np.array(Image.open(Path(r"film_agent\frames\throwing\CottonDiscus_0.jpg"))))
frame2_result = frame_node(np.array(Image.open(Path(r"film_agent\frames\not_throwing\20250401_163936.jpg"))))
schemes = [SchemeUnit(context=[action1_node, frame1_result], action=mps[0]), 
           SchemeUnit(context=[action2_node, frame2_result], action=mps[1]),
           SchemeUnit(context=[], action=mps[0])
          ]

pm = ProceduralMemory(schemes=schemes)
acs = [AttentionCodelet()]

lida_agent = {
'sensory_system': SensorySystem(pam=pam, sensory_memory=sm),
'csm': CurrentSituationalModel(),
'gw': GlobalWorkspace(attention_codelets=acs, broadcast_receivers=[pm]),
'procedural_system': ProceduralSystem(procedural_memory=pm),
'sensory_motor_system': SensoryMotorSystem(actuators=actuators, motor_plans=mps),
}


if __name__ == '__main__':
    # Initialize environment with reference image folders and reduced FPS for better playback
    try:
        env = FilmEnvironment(
            video_source=0,
            reference_image_folders=["throwing", "not_throwing"],
            output_dir=r"film_agent\test_data\LIDA",
            fps=30
        )
        
        try:
            # Get the total number of frames in the video
            total_frames = int(env.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing full video: {total_frames} frames")
            # Process all frames in the video
            minimally_conscious_agent(env, lida_agent, steps=total_frames)
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