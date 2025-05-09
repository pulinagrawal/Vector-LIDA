from film_agent.agent import vision_processor, compute_average_embedding, throwing_ref_node, other_ref_node
import torch

class BaseCLIPAgent:
    """
    Base class for all CLIP-based film agents
    """
    def __init__(self, environment):
        self.env = environment

        self.throwing_embeddings = self.env.collect_embeddings("throwing")
        self.not_throwing_embeddings = self.env.collect_embeddings("not_throwing")

        #self.throwing_embeddings = self.throwing_embeddings[1]
        #self.not_throwing_embeddings = self.not_throwing_embeddings[1]

        self.category_embeddings = {
            "throwing": compute_average_embedding(self.throwing_embeddings),
            "not_throwing": compute_average_embedding(self.not_throwing_embeddings)
        }

        throwing_ref_node.features = self.category_embeddings["throwing"]
        other_ref_node.features = self.category_embeddings["not_throwing"]

    def classify(self, frame, timestamp=None):
        current_features = vision_processor(frame)
        if not current_features:
            return None, 0.0

        current_node = current_features[0]
        throwing_sim = throwing_ref_node.similarity_function(throwing_ref_node, current_node)
        not_throwing_sim = other_ref_node.similarity_function(other_ref_node, current_node)
        #print(f"Throwing similarity: {throwing_sim}, Not throwing similarity: {not_throwing_sim}")

        confidence = max(throwing_sim, not_throwing_sim)
        decision = "throwing" if throwing_sim > not_throwing_sim else "not throwing"
        #print(f"Decision: {decision}, Confidence: {confidence:.4f}")
        return decision, confidence
