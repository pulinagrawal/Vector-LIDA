# LIDA Cognitive Architecture

## Overview
LIDA (Learning Intelligent Distribution Agent) models cognition through three repeating phases:
1. Understanding phase processes raw sensory data into structured Node objects.  
2. Consciousness phase selects highly activated coalitions for broadcast in the Global Workspace.  
3. Action Selection phase involves Procedural and SensoryMotor modules choosing and executing a behavior.

## Key Components
- **SensorySystem**: Converts sensory input into Nodes.  
- **CurrentSituationalModel (CSM)**: Maintains short-term context, receiving and updating nodes.  
- **GlobalWorkspace**: Competes coalitions to determine the content of consciousness.  
- **ProceduralSystem**: Chooses actions (Schemas) based on the winning coalition.  
- **SensoryMotorSystem**: Executes the chosen action.

## Extensibility & Flexibility
This implementation is built with modular Python classes, allowing:  
- Plug-and-play memory structures (e.g., different sensor processors).  
- Adaptable codelets (e.g., custom attention strategies).  
- Easy creation of specialized agents by substituting or extending modules.

## Usage
1. Import and initialize the modules you need.  
2. Run the cognitive cycle by providing an environment to the agent.  
3. Extend modules or override classes to create custom behavior.
