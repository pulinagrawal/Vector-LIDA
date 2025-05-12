import platform
import torch
import os
from torch import tensor

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


def compute_average_embedding(embeddings_list, ema_mode=True, prev_embedding=None, env=None):
    """Compute the average embedding from a list of embeddings
    
    Args:
        embeddings_list: List of embeddings to average
        ema_mode: If True, use exponential moving average instead of simple average
        prev_embedding: Previous EMA value (only used if ema_mode=True)
        env: FilmEnvironment instance to get EMA parameters from
        
    Returns:
        The averaged embedding (or EMA updated embedding if in EMA mode)
    """
    if not embeddings_list:
        return prev_embedding if ema_mode and prev_embedding is not None else None

    try:
        # Extract features if the objects have a 'features' attribute, otherwise use the objects directly
        features_list = [emb.features if hasattr(emb, 'features') else emb for emb in embeddings_list]
        
        if ema_mode and prev_embedding is not None:
            # Apply EMA update using the first embedding in the list
            # Get alpha from environment if provided, otherwise use default
            alpha = env.ema_alpha if env is not None else 0.1
            
            # Apply EMA formula: new_ema = alpha * current + (1 - alpha) * previous_ema
            current_features = tensor(features_list[0])
            prev_features = tensor(prev_embedding)
            
            # Make sure the dimensions match for the calculation
            if len(current_features.shape) == 1:
                current_features = current_features.unsqueeze(0)
            if len(prev_features.shape) == 1:
                prev_features = prev_features.unsqueeze(0)
                
            avg_embedding = alpha * current_features + (1 - alpha) * prev_features
        else:
            # Standard averaging
            features_list = tensor(features_list)
            avg_embedding = torch.mean(features_list, dim=0, keepdim=True)

        # Normalize the average embedding
        avg_embedding /= avg_embedding.norm(dim=-1, keepdim=True)
        
        return avg_embedding.squeeze(0).tolist()
    except Exception as e:
        print_error(f"Error computing average embedding: {e}")
        return None


# Define direct similarity function for nodes
def direct_cosine_similarity(node1, node2):
    """Calculate cosine similarity between two nodes directly using their features"""
    if (hasattr(node1, 'features') and node1.features is not None and 
        hasattr(node2, 'features') and node2.features is not None):
        try:
            similarity = torch.nn.functional.cosine_similarity(
                tensor(node1.features).unsqueeze(0),
                tensor(node2.features).unsqueeze(0)
            ).item()
            return similarity
        except Exception as e:
            print_error(f"Error calculating similarity: {e}")
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
