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


def compute_average_embedding(embeddings_list):
    """Compute the average embedding from a list of embeddings"""
    if not embeddings_list:
        return None

    try:
        # Extract features if the objects have a 'features' attribute, otherwise use the objects directly
        features_list = [emb.features if hasattr(emb, 'features') else emb for emb in embeddings_list]
        
        # Stack all embeddings and compute the mean
        stacked = torch.cat(features_list, dim=0)
        avg_embedding = torch.mean(stacked, dim=0, keepdim=True)

        # Normalize the average embedding
        avg_embedding /= avg_embedding.norm(dim=-1, keepdim=True)
        
        return avg_embedding
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
                node1.features, node2.features
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
