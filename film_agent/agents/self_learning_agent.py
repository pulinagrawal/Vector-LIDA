from film_agent.agents.base_agent import BaseCLIPAgent
from film_agent.agent import vision_processor, compute_average_embedding
import torch
import numpy as np
import random

class SelfLearningAgent(BaseCLIPAgent):
    """
    A self-learning CLIP agent that doesn't rely on labeled reference embeddings
    but instead learns categories through clustering
    """
    def __init__(self, environment):
        super().__init__(environment)
        
        # Discard the initial reference embeddings
        print("SelfLearningAgent: Discarding initial reference embeddings")
        
        # Initialize empty embedding clusters
        self.clusters = {
            "throwing": [],
            "not_throwing": []
        }
        
        # Create agent-specific reference nodes
        self.throwing_ref_node = None
        self.not_throwing_ref_node = None
        
        # Track frames added to each category
        self.frames_added = {
            "throwing": 0,
            "not_throwing": 0
        }
        
        # Clustering parameters
        self.similarity_threshold = 0.6  # Threshold for considering frames similar
        self.bootstrapping_frames = 20   # Number of frames to process before making classifications
        self.current_frame_index = 0
        
        # "Throwing" tends to have more motion - we'll use this as initial bias
        self.motion_threshold = 0.05
        self.last_features = None
        
        print("SelfLearningAgent initialized - will learn classifications from video content")
    
    def classify(self, frame, timestamp=None):
        """
        Classify the current frame based on learned clusters from the video itself.
        
        Initial frames are used for bootstrapping clusters without classification.
        """
        # Extract features from current frame
        current_features = vision_processor(frame)
        if not current_features:
            return None, 0.0
        
        current_node = current_features[0]
        self.current_frame_index += 1
        
        # Bootstrapping phase: cluster the first N frames without making classification decisions
        if self.current_frame_index <= self.bootstrapping_frames:
            self._bootstrap_clustering(current_node)
            # During bootstrapping, we'll just return a placeholder
            return "initializing", 0.0
        
        # After bootstrapping, we should have initial clusters
        # We can now classify based on similarity to cluster centers
        decision, confidence = self._classify_with_clusters(current_node)
        
        # Update clusters with the current frame based on the classification
        self._update_clusters(current_node, decision)
            
        return decision, confidence
    
    def _bootstrap_clustering(self, current_node):
        """
        Initialize clusters from the first few frames based on motion heuristics
        """
        if self.last_features is not None:
            # Calculate motion as feature difference from previous frame
            motion_magnitude = torch.norm(
                current_node.features - self.last_features, p=2
            ).item()
            
            # Assign to clusters based on motion
            if motion_magnitude > self.motion_threshold:
                self.clusters["throwing"].append(current_node)
                print(f"Bootstrap: Added frame to 'throwing' based on motion: {motion_magnitude:.4f}")
            else:
                self.clusters["not_throwing"].append(current_node)
                print(f"Bootstrap: Added frame to 'not_throwing' based on motion: {motion_magnitude:.4f}")
        else:
            # First frame goes to 'not_throwing' by default
            self.clusters["not_throwing"].append(current_node)
            print("Bootstrap: First frame added to 'not_throwing'")
            
        # Update last features
        self.last_features = current_node.features.clone()
        
        # If this is the last bootstrapping frame, compute initial cluster centers
        if self.current_frame_index == self.bootstrapping_frames:
            self._compute_cluster_centers()
            print(f"Bootstrapping complete - throwing: {len(self.clusters['throwing'])} frames, not_throwing: {len(self.clusters['not_throwing'])} frames")
    
    def _compute_cluster_centers(self):
        """
        Compute the centers of each cluster based on current frames
        """
        if self.clusters["throwing"]:
            throwing_embedding = compute_average_embedding(self.clusters["throwing"])
            if throwing_embedding is not None:
                self.throwing_ref_node = throwing_embedding
        
        if self.clusters["not_throwing"]:
            not_throwing_embedding = compute_average_embedding(self.clusters["not_throwing"])
            if not_throwing_embedding is not None:
                self.not_throwing_ref_node = not_throwing_embedding
    
    def _classify_with_clusters(self, current_node):
        """
        Classify a frame based on similarity to cluster centers
        """
        # If we don't have both cluster centers yet, make a random guess
        if self.throwing_ref_node is None or self.not_throwing_ref_node is None:
            return "not_throwing", 0.0
        
        # Calculate similarities to cluster centers
        throwing_sim = torch.nn.functional.cosine_similarity(
            self.throwing_ref_node, current_node.features
        ).item()
        
        not_throwing_sim = torch.nn.functional.cosine_similarity(
            self.not_throwing_ref_node, current_node.features
        ).item()
        
        # Calculate confidence as the absolute difference
        confidence = max(throwing_sim,not_throwing_sim)
        
        # Determine classification based on highest similarity
        if throwing_sim > not_throwing_sim:
            return "throwing", confidence
        else:
            return "not_throwing", confidence
    
    def _update_clusters(self, current_node, decision):
        """
        Update clusters based on the classification decision
        """
        # Add the current node to the appropriate cluster
        self.clusters[decision].append(current_node)
        self.frames_added[decision] += 1
        
        # Periodically recalculate cluster centers (every 10 frames)
        if (self.frames_added["throwing"] + self.frames_added["not_throwing"]) % 10 == 0:
            self._compute_cluster_centers()
            
            # Log the size of each cluster
            print(f"Updated clusters - throwing: {self.frames_added['throwing']} frames, "
                  f"not_throwing: {self.frames_added['not_throwing']} frames")