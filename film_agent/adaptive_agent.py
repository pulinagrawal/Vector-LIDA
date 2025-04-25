from base_agent import BaseCLIPAgent
from agent import compute_average_embedding, vision_processor, throwing_ref_node, other_ref_node

CONFIDENCE_THRESHOLD = 0.2

class AdaptiveCLIPAgent(BaseCLIPAgent):
    def classify(self, frame, timestamp=None):
        decision, confidence = super().classify(frame, timestamp)

        if decision is None:
            return None, 0.0

        current_node = vision_processor(frame)[0]

        if confidence > CONFIDENCE_THRESHOLD:
            if decision == "throwing":
                self.throwing_embeddings.append(current_node)
                self.category_embeddings["throwing"] = compute_average_embedding(self.throwing_embeddings)
                throwing_ref_node.features = self.category_embeddings["throwing"]
            else:
                self.not_throwing_embeddings.append(current_node)
                self.category_embeddings["not_throwing"] = compute_average_embedding(self.not_throwing_embeddings)
                other_ref_node.features = self.category_embeddings["not_throwing"]

        return decision, confidence
