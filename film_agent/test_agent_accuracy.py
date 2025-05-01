import json
import time
import cv2
from pathlib import Path
import argparse
from datetime import datetime

from agent import FilmEnvironment
from static_agent import StaticCLIPAgent
from adaptive_agent import AdaptiveCLIPAgent

def get_label_for_timestamp(timestamp, label_intervals):
    for entry in label_intervals:
        if entry["start"] <= timestamp < entry["end"]:
            return entry["label"]
    return "not throwing"

def run_test(video_path, ground_truth_path, agent_type, output_json="results.json"):
    # Generate a unique output filename by appending a timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = output_json.replace(".json", f"_{timestamp_str}.json")

    with open(ground_truth_path, 'r') as f:
        label_intervals = json.load(f)

    start_time = time.time()
    env = FilmEnvironment(video_path, reference_image_folders=["throwing", "not_throwing"], fps=30, display_frames=False)
    env_init_time = time.time() - start_time
    print(f"Environment initialization time: {env_init_time:.2f} seconds")

    start_time = time.time()
    if agent_type == "adaptive":
        agent = AdaptiveCLIPAgent(env)
    elif agent_type == "static":
        agent = StaticCLIPAgent(env)
    else:
        raise ValueError("Unsupported agent type. Use 'static' or 'adaptive'.")
    agent_init_time = time.time() - start_time
    print(f"Agent initialization time: {agent_init_time:.2f} seconds")

    total_frames = int(env.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = env.cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps}")
    results = {
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
        decision, _ = agent.classify(frame, timestamp)
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

        # Total frame processing time
        frame_processing_times.append(sensory_time + classification_time + logging_time)

        if frame_num > 0 and frame_num % 100 == 0:
            current_accuracy = results["correct_frames"] / results["total_frames"]
            avg_frame_time = sum(frame_processing_times) / len(frame_processing_times)
            print(f"Frame {frame_num}: Current Accuracy = {current_accuracy:.2%}, Avg Frame Time = {avg_frame_time:.2f} seconds")
            print(f"Breakdown: Sensory Time = {sensory_time:.2f}, Classification Time = {classification_time:.2f}, Logging Time = {logging_time:.2f}")

    results["accuracy"] = results["correct_frames"] / results["total_frames"] if results["total_frames"] > 0 else 0

    with open(output_json, 'w') as out:
        json.dump(results, out, indent=2)

    print(f"Done. Results written to {output_json} (Accuracy: {results['accuracy']:.2%})")
    print(f"Average frame processing time: {sum(frame_processing_times) / len(frame_processing_times):.2f} seconds")
    print(f"Breakdown: Avg Sensory Time = {sum(sensory_times) / len(sensory_times):.2f} seconds, "
          f"Avg Classification Time = {sum(classification_times) / len(classification_times):.2f} seconds, "
          f"Avg Logging Time = {sum(logging_times) / len(logging_times):.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--labels", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", default="results.json", help="Path to output results JSON")
    parser.add_argument("--agent", choices=["static", "adaptive"], default="adaptive", help="Agent type to use")
    args = parser.parse_args()

    run_test(args.video, args.labels, args.agent, args.output)
