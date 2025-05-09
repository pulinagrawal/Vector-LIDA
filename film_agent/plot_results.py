import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_metadata_from_filename(filename):
    """Extract metadata from the filename structure"""
    base = os.path.basename(filename)
    parts = base.split('_')
    metadata = {}
    
    if 'static' in base:
        metadata['agent_type'] = 'static'
    elif 'adaptive' in base:
        metadata['agent_type'] = 'adaptive'
    elif 'ema' in base:
        metadata['agent_type'] = 'ema'
    elif 'selflearning' in base:
        metadata['agent_type'] = 'selflearning'
    
    if 'with_ref' in base:
        metadata['use_initial_embeddings'] = True
    elif 'no_ref' in base:
        metadata['use_initial_embeddings'] = False
    
    # Try to extract parameters like conf, alpha, etc.
    for part in parts:
        if part.startswith('conf'):
            try:
                metadata['confidence_threshold'] = float(part[4:])
            except ValueError:
                pass
        elif part.startswith('alpha'):
            try:
                metadata['ema_alpha'] = float(part[5:])
            except ValueError:
                pass
        elif part.startswith('motion'):
            try:
                metadata['motion_threshold'] = float(part[6:])
            except ValueError:
                pass
        elif part.startswith('boot'):
            try:
                metadata['bootstrapping_frames'] = int(part[4:])
            except ValueError:
                pass
    
    return metadata

def load_results(results_dir):
    """Load all JSON results from a directory"""
    results = []
    for json_file in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Add filename to data for reference
                data['filename'] = os.path.basename(json_file)
                
                # If metadata is missing or incomplete, try to extract from filename
                if 'metadata' not in data:
                    data['metadata'] = extract_metadata_from_filename(json_file)
                else:
                    # Fill in any missing metadata from filename
                    filename_metadata = extract_metadata_from_filename(json_file)
                    for key, value in filename_metadata.items():
                        if key not in data['metadata'] or data['metadata'][key] is None:
                            data['metadata'][key] = value
                
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def plot_overall_accuracy(results, output_path=None, title=None):
    """Plot overall accuracy comparison between different agents"""
    agent_types = set()
    static_results = []
    adaptive_with_ref_results = []
    adaptive_no_ref_results = []
    ema_with_ref_results = []
    ema_no_ref_results = []
    selflearning_results = []
    
    # Group results by agent type and embedding usage
    for result in results:
        agent_type = result['metadata'].get('agent_type')
        use_initial_embeddings = result['metadata'].get('use_initial_embeddings')
        agent_types.add(agent_type)
        
        if agent_type == 'static':
            static_results.append(result)
        elif agent_type == 'adaptive':
            if use_initial_embeddings:
                adaptive_with_ref_results.append(result)
            else:
                adaptive_no_ref_results.append(result)
        elif agent_type == 'ema':
            if use_initial_embeddings:
                ema_with_ref_results.append(result)
            else:
                ema_no_ref_results.append(result)
        elif agent_type == 'selflearning':
            selflearning_results.append(result)
    
    # Prepare data for plotting
    categories = []
    accuracy_values = []
    colors = []
    
    # Add static (baseline)
    if static_results:
        categories.append("Static")
        accuracy_values.append(np.mean([r['accuracy'] for r in static_results]))
        colors.append('gray')
    
    # Add adaptive with reference embeddings
    if adaptive_with_ref_results:
        categories.append("Adaptive\n(with ref)")
        accuracy_values.append(np.mean([r['accuracy'] for r in adaptive_with_ref_results]))
        colors.append('blue')
    
    # Add adaptive without reference embeddings
    if adaptive_no_ref_results:
        categories.append("Adaptive\n(no ref)")
        accuracy_values.append(np.mean([r['accuracy'] for r in adaptive_no_ref_results]))
        colors.append('lightblue')
    
    # Add EMA with reference embeddings
    if ema_with_ref_results:
        categories.append("EMA\n(with ref)")
        accuracy_values.append(np.mean([r['accuracy'] for r in ema_with_ref_results]))
        colors.append('green')
    
    # Add EMA without reference embeddings
    if ema_no_ref_results:
        categories.append("EMA\n(no ref)")
        accuracy_values.append(np.mean([r['accuracy'] for r in ema_no_ref_results]))
        colors.append('lightgreen')
    
    # Add self-learning
    if selflearning_results:
        categories.append("Self-Learning")
        accuracy_values.append(np.mean([r['accuracy'] for r in selflearning_results]))
        colors.append('orange')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, accuracy_values, color=colors)
    plt.ylabel('Accuracy')
    if title:
        plt.title(title)
    else:
        plt.title('Accuracy Comparison by Agent Type')
    
    # Add exact values on top of bars
    for bar, value in zip(bars, accuracy_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2%}", 
                 ha='center', va='bottom')
    
    plt.ylim(0, max(accuracy_values) * 1.2)  # Add some space above the bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    return plt.gcf()

def plot_parameter_impact(results, parameter_name, output_path=None, title=None):
    """Plot the impact of a specific parameter on accuracy"""
    # Filter results that have the specified parameter
    filtered_results = [r for r in results if parameter_name in r['metadata'] and r['metadata'][parameter_name] is not None]
    
    if not filtered_results:
        print(f"No results found with parameter '{parameter_name}'")
        return None
    
    # Group by agent type and parameter value
    agent_types = set()
    param_values = set()
    data = {}
    
    for result in filtered_results:
        agent_type = result['metadata'].get('agent_type')
        param_value = result['metadata'].get(parameter_name)
        agent_types.add(agent_type)
        param_values.add(param_value)
        
        key = (agent_type, param_value)
        if key not in data:
            data[key] = []
        data[key].append(result['accuracy'])
    
    # Sort parameter values
    param_values = sorted(list(param_values))
    agent_types = sorted(list(agent_types))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Pick colors for each agent type
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_types)))
    
    for i, agent_type in enumerate(agent_types):
        agent_accuracies = []
        for param_value in param_values:
            key = (agent_type, param_value)
            if key in data:
                agent_accuracies.append(np.mean(data[key]))
            else:
                agent_accuracies.append(None)  # No data for this combination
        
        # Plot only if we have data for this agent type
        valid_indices = [i for i, x in enumerate(agent_accuracies) if x is not None]
        if valid_indices:
            valid_params = [param_values[i] for i in valid_indices]
            valid_accuracies = [agent_accuracies[i] for i in valid_indices]
            plt.plot(valid_params, valid_accuracies, 'o-', label=agent_type, color=colors[i])
            
            # Add values as text
            for x, y in zip(valid_params, valid_accuracies):
                plt.text(x, y+0.02, f"{y:.2%}", ha='center', va='bottom')
    
    param_display_name = {
        'confidence_threshold': 'Confidence Threshold',
        'ema_alpha': 'EMA Alpha',
        'motion_threshold': 'Motion Threshold',
        'bootstrapping_frames': 'Bootstrapping Frames'
    }.get(parameter_name, parameter_name)
    
    plt.xlabel(param_display_name)
    plt.ylabel('Accuracy')
    if title:
        plt.title(title)
    else:
        plt.title(f'Impact of {param_display_name} on Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    return plt.gcf()

def plot_accuracy_over_time(results, output_path=None, title=None, max_agents=5):
    """Plot accuracy over time (frames) to show learning progress"""
    # Find results with frame-by-frame logs
    results_with_logs = [r for r in results if 'log' in r and r['log']]
    
    if not results_with_logs:
        print("No results with frame logs found")
        return None
    
    # Sort by agent type and select up to max_agents to avoid overcluttering
    results_with_logs = sorted(results_with_logs, key=lambda r: (r['metadata'].get('agent_type', ''), r['metadata'].get('filename', '')))
    if len(results_with_logs) > max_agents:
        print(f"More than {max_agents} results with logs found. Selecting the first {max_agents} for clarity.")
        results_with_logs = results_with_logs[:max_agents]
    
    plt.figure(figsize=(12, 6))
    
    window_size = 100  # Window size for moving average
    
    for idx, result in enumerate(results_with_logs):
        agent_type = result['metadata'].get('agent_type', 'unknown')
        use_initial_embeddings = result['metadata'].get('use_initial_embeddings')
        
        # Create label
        if use_initial_embeddings is not None:
            embeddings_str = "with ref" if use_initial_embeddings else "no ref"
            label = f"{agent_type} ({embeddings_str})"
        else:
            label = agent_type
            
        # Extract additional parameters for the label
        params = []
        for param_name in ['confidence_threshold', 'ema_alpha', 'motion_threshold']:
            if param_name in result['metadata'] and result['metadata'][param_name] is not None:
                param_value = result['metadata'][param_name]
                params.append(f"{param_name.split('_')[0]}={param_value}")
        
        if params:
            label += f" ({', '.join(params)})"
        
        # Calculate rolling accuracy
        frames = []
        accuracies = []
        rolling_correct = 0
        
        for i, entry in enumerate(result['log']):
            if entry.get('correct') is not None:
                frames.append(entry.get('frame', i))
                rolling_correct += 1 if entry['correct'] else 0
                
                # Calculate accuracy over the window
                window_start = max(0, i - window_size + 1)
                window_count = i - window_start + 1
                window_correct = sum(1 for j in range(window_start, i + 1) if result['log'][j].get('correct', False))
                window_accuracy = window_correct / window_count if window_count > 0 else 0
                
                accuracies.append(window_accuracy)
        
        # Plot
        color = plt.cm.tab10(idx % 10)
        plt.plot(frames, accuracies, label=label, alpha=0.8, linewidth=2, color=color)
    
    plt.xlabel('Frame Number')
    plt.ylabel(f'Accuracy (Moving Window: {window_size} frames)')
    if title:
        plt.title(title)
    else:
        plt.title('Learning Progress: Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Plot results from agent tests')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save plots (defaults to results_dir/plots)')
    parser.add_argument('--batch_name', type=str, default=None, help='Name to include in plot titles and filenames')
    parser.add_argument('--compare_param', type=str, default=None, choices=['confidence_threshold', 'ema_alpha', 'motion_threshold', 'bootstrapping_frames'], 
                        help='Parameter to compare across agents')
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'plots')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Create title prefix based on batch name
    title_prefix = f"{args.batch_name} - " if args.batch_name else ""
    
    # Plot overall accuracy comparison
    output_path = os.path.join(args.output_dir, 'overall_accuracy.png')
    plot_overall_accuracy(results, output_path, title=f"{title_prefix}Agent Accuracy Comparison")
    
    # Plot parameter impact if requested
    if args.compare_param:
        output_path = os.path.join(args.output_dir, f'{args.compare_param}_impact.png')
        plot_parameter_impact(results, args.compare_param, output_path, 
                            title=f"{title_prefix}Impact of {args.compare_param} on Accuracy")
    else:
        # Generate all parameter comparisons
        for param in ['confidence_threshold', 'ema_alpha', 'motion_threshold', 'bootstrapping_frames']:
            output_path = os.path.join(args.output_dir, f'{param}_impact.png')
            plot_parameter_impact(results, param, output_path, 
                                title=f"{title_prefix}Impact of {param} on Accuracy")
    
    # Plot accuracy over time (learning progress)
    output_path = os.path.join(args.output_dir, 'learning_progress.png')
    plot_accuracy_over_time(results, output_path, title=f"{title_prefix}Learning Progress Over Time")
    
    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()