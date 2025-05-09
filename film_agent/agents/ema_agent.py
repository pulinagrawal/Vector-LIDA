from film_agent.agents.base_agent import BaseCLIPAgent
from film_agent.agent import vision_processor, throwing_ref_node, other_ref_node
import torch

class EMAAgent(BaseCLIPAgent):
    """
    An adaptive CLIP agent that uses Exponential Moving Average (EMA) to update embeddings,
    giving more weight to recent frames than older ones.
    """
    # Class constants that can be modified via command line
    CONFIDENCE_THRESHOLD = 0.3  # Confidence threshold for accepting new frames into the average
    EMA_ALPHA = 0.1  # EMA decay factor (higher = more weight to recent frames)
    
    def __init__(self, environment, use_initial_embeddings=True):
        super().__init__(environment)
        
        # Track frames added to each category
        self.frames_added = {
            "throwing": 0,
            "not_throwing": 0
        }
        
        # Option to discard initial embeddings
        self.use_initial_embeddings = use_initial_embeddings
        
        if use_initial_embeddings:
            # Store the initial embeddings as tensors for EMA updates
            self.throwing_ema = self.category_embeddings["throwing"].clone() if self.category_embeddings["throwing"] is not None else None
            self.not_throwing_ema = self.category_embeddings["not_throwing"].clone() if self.category_embeddings["not_throwing"] is not None else None
            print("EMAAgent initialized with starting reference embeddings")
        else:
            print("EMAAgent: Discarding initial reference embeddings")
            # Initialize with no embeddings
            self.throwing_ema = None
            self.not_throwing_ema = None
            
            # Set up bootstrapping parameters
            self.bootstrapping_frames = 20
            self.current_frame_index = 0
            self.last_features = None
            self.motion_threshold = 0.05
    
    def classify(self, frame, timestamp=None):
        """
        Classify the current frame and update embeddings with EMA if confidence is high enough.
        
        Args:
            frame: The current video frame
            timestamp: Optional timestamp of the frame
            
        Returns:
            Tuple of (decision, confidence)
        """
        # Get current features
        current_features = vision_processor(frame)
        if not current_features:
            return None, 0.0
        
        current_node = current_features[0]
        
        # If not using initial embeddings, handle bootstrapping phase
        if not self.use_initial_embeddings:
            self.current_frame_index += 1
            
            # During bootstrapping phase, build initial EMA values
            if self.current_frame_index <= self.bootstrapping_frames:
                if self.last_features is not None:
                    # Calculate motion as feature difference from previous frame
                    motion_magnitude = torch.norm(
                        current_node.features - self.last_features, p=2
                    ).item()
                    
                    # Assign to categories based on motion
                    if motion_magnitude > self.motion_threshold:
                        # Update throwing EMA
                        if self.throwing_ema is None:
                            self.throwing_ema = current_node.features.clone()
                        else:
                            self.throwing_ema = EMAAgent.EMA_ALPHA * current_node.features + (1 - EMAAgent.EMA_ALPHA) * self.throwing_ema
                        self.frames_added["throwing"] += 1
                        print(f"Bootstrap: Updated 'throwing' EMA based on motion: {motion_magnitude:.4f}")
                    else:
                        # Update not_throwing EMA
                        if self.not_throwing_ema is None:
                            self.not_throwing_ema = current_node.features.clone()
                        else:
                            self.not_throwing_ema = EMAAgent.EMA_ALPHA * current_node.features + (1 - EMAAgent.EMA_ALPHA) * self.not_throwing_ema
                        self.frames_added["not_throwing"] += 1
                        print(f"Bootstrap: Updated 'not_throwing' EMA based on motion: {motion_magnitude:.4f}")
                else:
                    # First frame goes to 'not_throwing' by default
                    self.not_throwing_ema = current_node.features.clone()
                    self.frames_added["not_throwing"] += 1
                    print("Bootstrap: First frame used to initialize 'not_throwing' EMA")
                    
                # Update last features
                self.last_features = current_node.features.clone()
                
                if self.current_frame_index == self.bootstrapping_frames:
                    print(f"Bootstrapping complete - throwing frames: {self.frames_added['throwing']}, not_throwing frames: {self.frames_added['not_throwing']}")
                    
                return "initializing", 0.0
        
        # Use available EMA values for classification
        decision, confidence = None, 0.0
        
        # Only classify if we have both category embeddings
        if self.throwing_ema is not None and self.not_throwing_ema is not None:
            # Calculate similarities
            if hasattr(current_node, 'features') and current_node.features is not None:
                # Fix: Ensure proper tensor dimensions for similarity calculation
                current_features = current_node.features
                
                # Make sure tensors have the right shape for cosine similarity
                if len(current_features.shape) == 2:
                    current_features_reshaped = current_features
                else:
                    # Add batch dimension if needed
                    current_features_reshaped = current_features.unsqueeze(0)
                
                throwing_features = self.throwing_ema.unsqueeze(0) if len(self.throwing_ema.shape) == 1 else self.throwing_ema
                not_throwing_features = self.not_throwing_ema.unsqueeze(0) if len(self.not_throwing_ema.shape) == 1 else self.not_throwing_ema
                
                # Debug prints to help diagnose shape issues
                # print(f"Current features shape: {current_features_reshaped.shape}")
                # print(f"Throwing features shape: {throwing_features.shape}")
                # print(f"Not throwing features shape: {not_throwing_features.shape}")
                
                # Calculate similarities ensuring batch dimension compatibility
                throwing_sim = torch.nn.functional.cosine_similarity(
                    throwing_features, current_features_reshaped, dim=1
                ).item()
                
                not_throwing_sim = torch.nn.functional.cosine_similarity(
                    not_throwing_features, current_features_reshaped, dim=1
                ).item()
                
                confidence = max(throwing_sim,not_throwing_sim)
                decision = "throwing" if throwing_sim > not_throwing_sim else "not throwing"
        else:
            # Default decision if embeddings aren't available yet
            decision = "not throwing"
            confidence = 0.0

        # Update embeddings if confidence exceeds threshold
        if confidence > EMAAgent.CONFIDENCE_THRESHOLD:
            self._update_embeddings_ema(current_node, decision)
            
        return decision, confidence
    
    def _update_embeddings_ema(self, current_node, decision):
        """
        Update the category embeddings using Exponential Moving Average (EMA).
        
        EMA = alpha * current_value + (1 - alpha) * previous_EMA
        
        Args:
            current_node: The node with features from the current frame
            decision: The classification decision ("throwing" or "not throwing")
        """
        if not hasattr(current_node, 'features') or current_node.features is None:
            return
            
        # Update the appropriate embedding based on decision
        if decision == "throwing":
            self.frames_added["throwing"] += 1
            
            # Apply EMA formula to update the throwing embedding
            if self.throwing_ema is not None:
                self.throwing_ema = EMAAgent.EMA_ALPHA * current_node.features + (1 - EMAAgent.EMA_ALPHA) * self.throwing_ema
            else:
                self.throwing_ema = current_node.features.clone()
                
            # Only update global reference node if we're using initial embeddings
            if self.use_initial_embeddings:
                throwing_ref_node.features = self.throwing_ema.unsqueeze(0)
            
            # Print update info occasionally
            if self.frames_added["throwing"] % 10 == 0:
                print(f"Updated 'throwing' embedding with EMA (alpha={EMAAgent.EMA_ALPHA}). Total frames: {self.frames_added['throwing']}")
                
        else:  # "not throwing"
            self.frames_added["not_throwing"] += 1
            
            # Apply EMA formula to update the not_throwing embedding
            if self.not_throwing_ema is not None:
                self.not_throwing_ema = EMAAgent.EMA_ALPHA * current_node.features + (1 - EMAAgent.EMA_ALPHA) * self.not_throwing_ema
            else:
                self.not_throwing_ema = current_node.features.clone()
                
            # Only update global reference node if we're using initial embeddings
            if self.use_initial_embeddings:
                other_ref_node.features = self.not_throwing_ema.unsqueeze(0)
            
            # Print update info occasionally
            if self.frames_added["not_throwing"] % 10 == 0:
                print(f"Updated 'not_throwing' embedding with EMA (alpha={EMAAgent.EMA_ALPHA}). Total frames: {self.frames_added['not_throwing']}")