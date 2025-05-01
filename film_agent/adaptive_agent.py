from base_agent import BaseCLIPAgent
from agent import compute_average_embedding, vision_processor, throwing_ref_node, other_ref_node

CONFIDENCE_THRESHOLD = 0.03

class AdaptiveCLIPAgent(BaseCLIPAgent):
    def classify(self, frame, timestamp=None):
        # Original classification
        decision, confidence = super().classify(frame, timestamp)

        if decision is None:
            return None, 0.0

        # New classification using average embedding
        current_node = vision_processor(frame)[0]
        avg_decision, avg_confidence = self._classify_with_average_embedding(current_node)

        # Use the classification with the higher confidence
        #print(f"Average Confidence: {avg_confidence}, Original Confidence: {confidence}")
        if avg_confidence > confidence:
            decision, confidence = avg_decision, avg_confidence

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

    def _classify_with_average_embedding(self, current_node):
        """
        Classify the current node using the average embeddings for 'throwing' and 'not_throwing'.
        """
        throwing_similarity = current_node.similarity(self.category_embeddings["throwing"])
        not_throwing_similarity = current_node.similarity(self.category_embeddings["not_throwing"])

        if throwing_similarity > not_throwing_similarity:
            return "throwing", throwing_similarity
        else:
            return "not_throwing", not_throwing_similarity
