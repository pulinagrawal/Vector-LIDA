import json
import time
import os
import cv2
from pathlib import Path
import argparse
from datetime import datetime

from agent import FilmEnvironment
# Update imports to use the agents package
from agents.static_agent import StaticCLIPAgent
from agents.adaptive_agent import AdaptiveCLIPAgent
from agents.ema_agent import EMAAgent
from agents.self_learning_agent import SelfLearningAgent

def get_label_for_timestamp(timestamp, label_intervals):
    for entry in label_intervals:
        if entry["start"] <= timestamp < entry["end"]:
            return entry["label"]
    return "not throwing"

def run_test(video_path, ground_truth_path, agent_type, output_json="results.json", 
             display_video=False, use_initial_embeddings=True, 
             confidence_threshold=None, ema_alpha=None, motion_threshold=None,
             bootstrapping_frames=None, organize_by_folders=False):
    """
    Run a test with the specified agent and parameters
    
    Args:
        video_path: Path to the video file
        ground_truth_path: Path to ground truth labels
        agent_type: Type of agent to use (static, adaptive, ema, selflearning)
        output_json: Base path for output JSON
        display_video: Whether to display the video during processing
        use_initial_embeddings: Whether to use initial reference embeddings
        confidence_threshold: Confidence threshold for updating embeddings
        ema_alpha: Alpha value for EMA agent
        motion_threshold: Motion threshold for self-learning agents
        bootstrapping_frames: Number of frames to use for bootstrapping
        organize_by_folders: Whether to organize results in folders by agent type
    """
    # Generate a unique output filename with all parameters included
    video_name = Path(video_path).stem
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    embeddings_str = "with_ref" if use_initial_embeddings else "no_ref"
    
    # Create parameter string for filename
    params = []
    if confidence_threshold is not None:
        params.append(f"conf{confidence_threshold}")
    if ema_alpha is not None and agent_type == "ema":
        params.append(f"alpha{ema_alpha}")
    if motion_threshold is not None and (agent_type == "selflearning" or not use_initial_embeddings):
        params.append(f"motion{motion_threshold}")
    if bootstrapping_frames is not None and (agent_type == "selflearning" or not use_initial_embeddings):
        params.append(f"boot{bootstrapping_frames}")
    
    params_str = "_".join(params) if params else "default"
    
    # Create the output filename or folder structure
    if organize_by_folders:
        # Check if output_json has a specific folder path included
        if os.path.dirname(output_json):
            # Use the directory specified in output_json
            result_dir = Path(output_json)
            # Create the directory if it doesn't exist
            result_dir.parent.mkdir(parents=True, exist_ok=True)
            output_path = result_dir
        else:
            # Organize results in folders by agent type
            result_dir = Path("film_agent/test_data/results") / agent_type / embeddings_str / params_str
            result_dir.mkdir(parents=True, exist_ok=True)
            output_path = result_dir / f"{video_name}_{timestamp_str}.json"
    else:
        # Check if output_json has a specific folder path included
        output_base = Path(output_json)
        if output_base.parent != Path('.'):
            # Make sure the directory exists
            output_base.parent.mkdir(parents=True, exist_ok=True)
            # Use the specified path with our naming convention
            output_path = output_base.parent / f"{output_base.stem}_{agent_type}_{embeddings_str}_{params_str}_{video_name}_{timestamp_str}.json"
        else:
            # Use a long filename with all parameters in the current directory
            output_path = output_json.replace(".json", f"_{agent_type}_{embeddings_str}_{params_str}_{video_name}_{timestamp_str}.json")

    print(f"Results will be saved to: {output_path}")

    with open(ground_truth_path, 'r') as f:
        label_intervals = json.load(f)

    start_time = time.time()
    env = FilmEnvironment(video_path, reference_image_folders=["throwing", "not_throwing"], fps=30, display_frames=False)
    env_init_time = time.time() - start_time
    print(f"Environment initialization time: {env_init_time:.2f} seconds")

    start_time = time.time()
    
    # Create agent with specified parameters
    if agent_type == "adaptive":
        # Set the confidence threshold if specified
        if confidence_threshold is not None:
            AdaptiveCLIPAgent.CONFIDENCE_THRESHOLD = confidence_threshold
        
        # Set the motion threshold for bootstrapping if specified
        if motion_threshold is not None and not use_initial_embeddings:
            motion_param = motion_threshold
        else:
            motion_param = 0.05  # default
            
        # Set bootstrapping frames if specified
        if bootstrapping_frames is not None and not use_initial_embeddings:
            bootstrap_param = bootstrapping_frames
        else:
            bootstrap_param = 20  # default
            
        agent = AdaptiveCLIPAgent(env, use_initial_embeddings=use_initial_embeddings)
        
        # Set parameters after creation
        if not use_initial_embeddings:
            agent.motion_threshold = motion_param
            agent.bootstrapping_frames = bootstrap_param
            
    elif agent_type == "static":
        agent = StaticCLIPAgent(env)
    elif agent_type == "ema":
        # Set the confidence threshold if specified
        if confidence_threshold is not None:
            EMAAgent.CONFIDENCE_THRESHOLD = confidence_threshold
            
        # Set the alpha value if specified
        if ema_alpha is not None:
            EMAAgent.EMA_ALPHA = ema_alpha
            
        # Set the motion threshold for bootstrapping if specified
        if motion_threshold is not None and not use_initial_embeddings:
            motion_param = motion_threshold
        else:
            motion_param = 0.05  # default
            
        # Set bootstrapping frames if specified
        if bootstrapping_frames is not None and not use_initial_embeddings:
            bootstrap_param = bootstrapping_frames
        else:
            bootstrap_param = 20  # default
            
        agent = EMAAgent(env, use_initial_embeddings=use_initial_embeddings)
        
        # Set parameters after creation
        if not use_initial_embeddings:
            agent.motion_threshold = motion_param
            agent.bootstrapping_frames = bootstrap_param
            
    elif agent_type == "selflearning":
        agent = SelfLearningAgent(env)
        
        # Set the motion threshold if specified
        if motion_threshold is not None:
            agent.motion_threshold = motion_threshold
            
        # Set bootstrapping frames if specified
        if bootstrapping_frames is not None:
            agent.bootstrapping_frames = bootstrapping_frames
    else:
        raise ValueError("Unsupported agent type. Use 'static', 'adaptive', 'ema', or 'selflearning'.")
    
    agent_init_time = time.time() - start_time
    print(f"Agent initialization time: {agent_init_time:.2f} seconds")

    total_frames = int(env.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = env.cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Initialize display window if requested
    if display_video:
        cv2.namedWindow("Test Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Test Video", 800, 600)
    
    # Initialize results with complete parameter information
    results = {
        "metadata": {
            "agent_type": agent_type,
            "use_initial_embeddings": use_initial_embeddings,
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else 
                (AdaptiveCLIPAgent.CONFIDENCE_THRESHOLD if agent_type == "adaptive" else 
                EMAAgent.CONFIDENCE_THRESHOLD if agent_type == "ema" else None),
            "ema_alpha": ema_alpha if ema_alpha is not None else 
                (EMAAgent.EMA_ALPHA if agent_type == "ema" else None),
            "motion_threshold": motion_threshold if motion_threshold is not None else 
                (agent.motion_threshold if hasattr(agent, "motion_threshold") else None),
            "bootstrapping_frames": bootstrapping_frames if bootstrapping_frames is not None else
                (agent.bootstrapping_frames if hasattr(agent, "bootstrapping_frames") else None),
            "video": Path(video_path).name,
            "labels": Path(ground_truth_path).name,
            "test_date": timestamp_str
        },
        "accuracy": 0.0,
        "total_frames": 0,
        "correct_frames": 0,
        "log": []
    }

    frame_processing_times = []
    sensory_times = []
    classification_times = []
    logging_times = []

    for frame_num in range(total_frames):
        timestamp = frame_num / fps

        # Measure time for receiving sensory stimuli
        start_time = time.time()
        frame = env.receive_sensory_stimuli()
        sensory_time = time.time() - start_time
        sensory_times.append(sensory_time)

        if frame is None:
            break

        # Measure time for classification
        start_time = time.time()
        decision, confidence = agent.classify(frame, timestamp)
        classification_time = time.time() - start_time
        classification_times.append(classification_time)

        if decision is None:
            continue

        # Measure time for logging
        start_time = time.time()
        true_label = get_label_for_timestamp(timestamp, label_intervals)
        correct = (decision == true_label)

        results["total_frames"] += 1
        if correct:
            results["correct_frames"] += 1

        results["log"].append({
            "frame": frame_num,
            "timestamp": round(timestamp, 2),
            "agent_decision": decision,
            "true_label": true_label,
            "correct": correct
        })
        logging_time = time.time() - start_time
        logging_times.append(logging_time)

        # Display the frame with annotations if requested
        if display_video:
            # Create a copy of the frame for display
            display_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
            
            # Add annotations
            # Decision text (green if correct, red if wrong)
            color = (0, 255, 0) if correct else (0, 0, 255)
            cv2.putText(display_frame, f"Decision: {decision}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # True label
            cv2.putText(display_frame, f"True label: {true_label}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Confidence
            cv2.putText(display_frame, f"Confidence: {confidence:.4f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Frame info
            cv2.putText(display_frame, f"Frame: {frame_num}/{total_frames}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Accuracy so far
            current_accuracy = results["correct_frames"] / results["total_frames"] if results["total_frames"] > 0 else 0
            cv2.putText(display_frame, f"Accuracy: {current_accuracy:.2%}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Test Video", display_frame)
            
            # Process keyboard input (q or ESC to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("Video display interrupted by user")
                break

        # Total frame processing time
        frame_processing_times.append(sensory_time + classification_time + logging_time)

        if frame_num > 0 and frame_num % 100 == 0:
            current_accuracy = results["correct_frames"] / results["total_frames"]
            avg_frame_time = sum(frame_processing_times[-100:]) / min(100, len(frame_processing_times[-100:]))
            print(f"Frame {frame_num}/{total_frames}: Current Accuracy = {current_accuracy:.2%}, Avg Frame Time = {avg_frame_time:.2f} seconds")
            print(f"Breakdown: Sensory Time = {sensory_time:.2f}, Classification Time = {classification_time:.2f}, Logging Time = {logging_time:.2f}")

    results["accuracy"] = results["correct_frames"] / results["total_frames"] if results["total_frames"] > 0 else 0

    with open(output_path, 'w') as out:
        json.dump(results, out, indent=2)

    print(f"Done. Results written to {output_path} (Accuracy: {results['accuracy']:.2%})")
    print(f"Average frame processing time: {sum(frame_processing_times) / len(frame_processing_times):.2f} seconds")
    print(f"Breakdown: Avg Sensory Time = {sum(sensory_times) / len(sensory_times):.2f} seconds, "
          f"Avg Classification Time = {sum(classification_times) / len(classification_times):.2f} seconds, "
          f"Avg Logging Time = {sum(logging_times) / len(logging_times):.2f} seconds")
          
    # Clean up display resources
    if display_video:
        cv2.destroyAllWindows()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--labels", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", default="results.json", help="Path to output results JSON")
    parser.add_argument("--agent", choices=["static", "adaptive", "ema", "selflearning"], default="adaptive", 
                        help="Agent type to use")
    parser.add_argument("--display", action="store_true", help="Display video while testing")
    parser.add_argument("--use_initial_embeddings", action="store_true", help="Use initial embeddings for the agent")
    
    # New parameter arguments
    parser.add_argument("--confidence_threshold", type=float, help="Confidence threshold for updating embeddings")
    parser.add_argument("--ema_alpha", type=float, help="Alpha value for EMA agent (higher = more weight to recent frames)")
    parser.add_argument("--motion_threshold", type=float, help="Motion threshold for self-learning agents")
    parser.add_argument("--bootstrapping_frames", type=int, help="Number of frames to use for bootstrapping")
    parser.add_argument("--organize_by_folders", action="store_true", 
                        help="Organize results in folders rather than using long filenames")

    args = parser.parse_args()

    run_test(
        args.video, 
        args.labels, 
        args.agent, 
        args.output, 
        display_video=args.display, 
        use_initial_embeddings=args.use_initial_embeddings,
        confidence_threshold=args.confidence_threshold,
        ema_alpha=args.ema_alpha,
        motion_threshold=args.motion_threshold,
        bootstrapping_frames=args.bootstrapping_frames,
        organize_by_folders=args.organize_by_folders
    )
