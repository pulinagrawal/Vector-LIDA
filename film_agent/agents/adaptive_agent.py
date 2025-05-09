from film_agent.agents.base_agent import BaseCLIPAgent
from film_agent.agent import compute_average_embedding, vision_processor, throwing_ref_node, other_ref_node
import torch

class AdaptiveCLIPAgent(BaseCLIPAgent):
    """
    An adaptive CLIP agent that starts with reference embeddings and updates them
    over time based on high-confidence classifications.
    """
    # Class constants that can be modified via command line
    CONFIDENCE_THRESHOLD = 0.5  # Threshold for accepting new frames into the average
    
    def __init__(self, environment, use_initial_embeddings=True):
        super().__init__(environment)
        
        # Track frames added to each category for more precise updates
        self.frames_added = {
            "throwing": 0,
            "not_throwing": 0
        }
        
        # Option to discard initial embeddings
        self.use_initial_embeddings = use_initial_embeddings
        
        # If not using initial embeddings, we'll need our own reference nodes
        if not use_initial_embeddings:
            print("AdaptiveCLIPAgent: Discarding initial reference embeddings")
            # Create empty lists for embeddings
            self.throwing_embeddings = []
            self.not_throwing_embeddings = []
            
            # Initialize with empty category embeddings
            self.category_embeddings = {
                "throwing": None,
                "not_throwing": None
            }
            
            # We'll need to bootstrap initial classifications
            self.bootstrapping_frames = 20
            self.current_frame_index = 0
            self.last_features = None
            self.motion_threshold = 0.05
        else:
            print("AdaptiveCLIPAgent initialized with starting reference embeddings")
    
    def classify(self, frame, timestamp=None):
        """
        Classify the current frame and update embeddings if confidence is high enough.
        
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
            
            # During bootstrapping phase, build initial clusters
            if self.current_frame_index <= self.bootstrapping_frames:
                if self.last_features is not None:
                    # Calculate motion as feature difference from previous frame
                    motion_magnitude = torch.norm(
                        current_node.features - self.last_features, p=2
                    ).item()
                    
                    # Assign to categories based on motion
                    if motion_magnitude > self.motion_threshold:
                        self.throwing_embeddings.append(current_node)
                        print(f"Bootstrap: Added frame to 'throwing' based on motion: {motion_magnitude:.4f}")
                    else:
                        self.not_throwing_embeddings.append(current_node)
                        print(f"Bootstrap: Added frame to 'not_throwing' based on motion: {motion_magnitude:.4f}")
                else:
                    # First frame goes to 'not_throwing' by default
                    self.not_throwing_embeddings.append(current_node)
                    print("Bootstrap: First frame added to 'not_throwing'")
                    
                # Update last features
                self.last_features = current_node.features.clone()
                
                # If this is the last bootstrapping frame, compute initial embeddings
                if self.current_frame_index == self.bootstrapping_frames:
                    if self.throwing_embeddings:
                        self.category_embeddings["throwing"] = compute_average_embedding(self.throwing_embeddings)
                    if self.not_throwing_embeddings:
                        self.category_embeddings["not_throwing"] = compute_average_embedding(self.not_throwing_embeddings)
                    print(f"Bootstrapping complete - throwing: {len(self.throwing_embeddings)} frames, not_throwing: {len(self.not_throwing_embeddings)} frames")
                
                return "initializing", 0.0
        
        # Use the base classification approach with available embeddings
        decision, confidence = None, 0.0
        
        # Only classify if we have both category embeddings
        if self.category_embeddings["throwing"] is not None and self.category_embeddings["not_throwing"] is not None:
            # Calculate similarities
            if hasattr(current_node, 'features') and current_node.features is not None:
                throwing_sim = torch.nn.functional.cosine_similarity(
                    self.category_embeddings["throwing"], current_node.features
                ).item()
                
                not_throwing_sim = torch.nn.functional.cosine_similarity(
                    self.category_embeddings["not_throwing"], current_node.features
                ).item()
                
                confidence = max(throwing_sim,not_throwing_sim)
                decision = "throwing" if throwing_sim > not_throwing_sim else "not throwing"
        else:
            # Default decision if embeddings aren't available yet
            decision = "not throwing"
            confidence = 0.0
        
        # Update embeddings if confidence exceeds threshold
        if confidence > AdaptiveCLIPAgent.CONFIDENCE_THRESHOLD:
            self._update_embeddings(current_node, decision)
            
        return decision, confidence
    
    def _update_embeddings(self, current_node, decision):
        """
        Update the category embeddings with a new frame.
        
        Args:
            current_node: The node with features from the current frame
            decision: The classification decision ("throwing" or "not throwing")
        """
        # Update the appropriate embedding list based on decision
        if decision == "throwing":
            self.throwing_embeddings.append(current_node)
            self.frames_added["throwing"] += 1
            
            # Recompute average embedding
            new_embedding = compute_average_embedding(self.throwing_embeddings)
            if new_embedding is not None:
                self.category_embeddings["throwing"] = new_embedding
                
                # Only update global reference node if we're using initial embeddings
                if self.use_initial_embeddings:
                    throwing_ref_node.features = new_embedding
                
                # Print update info occasionally
                if self.frames_added["throwing"] % 10 == 0:
                    print(f"Updated 'throwing' embedding with {self.frames_added['throwing']} total frames")
                
        else:  # "not throwing"
            self.not_throwing_embeddings.append(current_node)
            self.frames_added["not_throwing"] += 1
            
            # Recompute average embedding
            new_embedding = compute_average_embedding(self.not_throwing_embeddings)
            if new_embedding is not None:
                self.category_embeddings["not_throwing"] = new_embedding
                
                # Only update global reference node if we're using initial embeddings
                if self.use_initial_embeddings:
                    other_ref_node.features = new_embedding
                
                # Print update info occasionally
                if self.frames_added["not_throwing"] % 10 == 0:
                    print(f"Updated 'not_throwing' embedding with {self.frames_added['not_throwing']} total frames")
