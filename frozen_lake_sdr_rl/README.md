## Introduction

This will enable a set of experiments with a sparse distributed representations based RL agent for the frozen lake environment. These experiments will help in conceptualizing and validating the task dependent representations in place cells hypothesis. The brief of the hypothesis is that the place cells help in learning a task dependent representation. This agent will have a representation of the current map as a sparse distributed representation (SDR) that is randomly initialized for every unique map. This agent will represent the individual boxes in the 2D space like grid cells with sparse activations of a collection of cells. Concretly, there is a transformation from the SDR of the current location to the SDR of the current map. 

The information the agent currently has is the SDR of the locations around the agent and the features present at those locations. The locations around the agent can be represented by sensor locations that are fixed relative to each other and the agent.  

>The agent processes this local sensory information to build a consistent internal representation of the environment. Through experience, it learns to correlate specific sensor activations with positions in the global map. This is similar to how hippocampal place cells in mammals integrate sensory cues and movement information to form spatial representations. When the agent encounters a familiar pattern of features in its sensory field, it can infer its absolute position within the larger environment, even without explicit coordinates.
>
>The process works as follows:
>
>1. **Sensory Encoding**: The agent has sensory receptors that detect features in adjacent locations (water, ice, hole, goal). Each detected feature activates a specific subset of neurons in the sensory layer.
>
>2. **Position Inference**: Through a Sparse Distributed Representation (SDR) mechanism, the agent maintains an internal model where:
>  - Each possible position on the map has a unique SDR pattern (similar to place cells)
>  - These patterns are initially random but become associated with specific sensory inputs
>  - A sparse set (~2%) of neurons activates for each position
>
>3. **Pattern Matching**: When the agent observes a particular configuration of features in its sensory field, it performs pattern completion:
>  - The observed sensory pattern is projected to the position representation layer
>  - If the pattern matches a previously experienced location, the corresponding place cell ensemble activates
>  - The activation strength correlates with the confidence of position recognition
>
>4. **Update Mechanism**: As the agent moves, it:
>  - Predicts the expected new sensory input based on its action
>  - Compares actual sensory input with prediction
>  - Strengthens connections between sensory patterns and position representations when predictions are accurate
>  - Adjusts representations when predictions fail
>
>This creates a self-organizing map where the agent progressively builds a coherent spatial model without explicit coordinates.

Let D be the coding field that represents the displacement. Let L be the coding field that represents the locations in the map or the grid cell coding field with multiple modules. 
L tries to predict the next location based on the current location and the displacement D. Probably based on temporal pooling algorithm.

Now, it needs to fire the correct grid cells that uniquely identifies its position in the map. 

 NOTE: This is a simplification; The agent will have to transform the displacement SDRs corresponding to each action into the grid cell activations of each of the locations to obtain the SDR of the locations around it. But since they are one-to-one mappings, we can skip that step for now.



 The agent will have an SDR representation of each of the possible actions. The agent will not know where it is on the map to begin with. As it takes actions, it will update its SDR representation of the current state based on the action taken, the location it moved to. The agent will learn to associate certain SDR patterns with higher rewards over time, allowing it to navigate the frozen lake more effectively.