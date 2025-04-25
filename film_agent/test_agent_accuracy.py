import json
import time
import cv2
from pathlib import Path
import argparse

from agent import FilmEnvironment
from static_agent import StaticCLIPAgent
from adaptive_agent import AdaptiveCLIPAgent

def get_label_for_timestamp(timestamp, label_intervals):
    for entry in label_intervals:
        if entry["start"] <= timestamp < entry["end"]:
            return entry["label"]
    return "not throwing"

def run_test(video_path, ground_truth_path, agent_type, output_json="results.json"):
    with open(ground_truth_path, 'r') as f:
        label_intervals = json.load(f)

    env = FilmEnvironment(video_path, reference_image_folders=["throwing", "not_throwing"], fps=30, display_frames=False)

    if agent_type == "adaptive":
        agent = AdaptiveCLIPAgent(env)
    elif agent_type == "static":
        agent = StaticCLIPAgent(env)
    else:
        raise ValueError("Unsupported agent type. Use 'static' or 'adaptive'.")

    total_frames = int(env.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = env.cap.get(cv2.CAP_PROP_FPS)

    results = {
        "accuracy": 0.0,
        "total_frames": 0,
        "correct_frames": 0,
        "log": []
    }

    for frame_num in range(total_frames):
        timestamp = frame_num / fps
        frame = env.receive_sensory_stimuli()
        if frame is None:
            break

        decision, _ = agent.classify(frame, timestamp)
        if decision is None:
            continue

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

        if frame_num > 0 and frame_num % 100 == 0:
            current_accuracy = results["correct_frames"] / results["total_frames"]
            print(f"Frame {frame_num}: Current Accuracy = {current_accuracy:.2%}")

    results["accuracy"] = results["correct_frames"] / results["total_frames"] if results["total_frames"] > 0 else 0

    with open(output_json, 'w') as out:
        json.dump(results, out, indent=2)

    print(f"Done. Results written to {output_json} (Accuracy: {results['accuracy']:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--labels", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", default="results.json", help="Path to output results JSON")
    parser.add_argument("--agent", choices=["static", "adaptive"], default="adaptive", help="Agent type to use")
    args = parser.parse_args()

    run_test(args.video, args.labels, args.agent, args.output)
