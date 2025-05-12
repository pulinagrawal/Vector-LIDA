import os
import sys
import json
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up better styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
                try:
                    data = json.load(f)
                    
                    # Validate required fields
                    if 'accuracy' not in data:
                        print(f"Warning: Missing 'accuracy' in {json_file}")
                        # Add a None accuracy to allow processing but it will be filtered later
                        data['accuracy'] = None
                        
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
                    
                    # Ensure agent_type exists in metadata
                    if 'agent_type' not in data['metadata'] or data['metadata']['agent_type'] is None:
                        print(f"Warning: Could not determine agent_type for {json_file}")
                        data['metadata']['agent_type'] = "unknown"
                    
                    results.append(data)
                    
                except json.JSONDecodeError as je:
                    print(f"Invalid JSON format in {json_file}: {je}")
                    
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
    
    # Prepare data for plotting
    categories = []
    accuracy_values = []
    colors = []
    error_bars = []
      # Add static (baseline)
    if static_results:
        categories.append("Static")
        accuracies = [r['accuracy'] for r in static_results if r['accuracy'] is not None]
        if accuracies:  # Check if list is not empty
            accuracy_values.append(np.mean(accuracies))
            error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
            colors.append('#7f7f7f')  # Gray
        else:
            # Remove the category if no valid data
            categories.pop()
    
    # Add adaptive with reference embeddings
    if adaptive_with_ref_results:
        categories.append("Adaptive\n(with ref)")
        accuracies = [r['accuracy'] for r in adaptive_with_ref_results if r['accuracy'] is not None]
        if accuracies:  # Check if list is not empty
            accuracy_values.append(np.mean(accuracies))
            error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
            colors.append('#1f77b4')  # Blue
        else:
            # Remove the category if no valid data
            categories.pop()
    
    # Add adaptive without reference embeddings
    if adaptive_no_ref_results:
        categories.append("Adaptive\n(no ref)")
        accuracies = [r['accuracy'] for r in adaptive_no_ref_results if r['accuracy'] is not None]
        if accuracies:  # Check if list is not empty
            accuracy_values.append(np.mean(accuracies))
            error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
            colors.append('#aec7e8')  # Light blue
        else:
            # Remove the category if no valid data
            categories.pop()
      # Add EMA with reference embeddings
    if ema_with_ref_results:
        categories.append("EMA\n(with ref)")
        accuracies = [r['accuracy'] for r in ema_with_ref_results if r['accuracy'] is not None]
        if accuracies:  # Check if list is not empty
            accuracy_values.append(np.mean(accuracies))
            error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
            colors.append('#2ca02c')  # Green
        else:
            # Remove the category if no valid data
            categories.pop()
            
    # Add EMA without reference embeddings
    if ema_no_ref_results:
        categories.append("EMA\n(no ref)")
        accuracies = [r['accuracy'] for r in ema_no_ref_results if r['accuracy'] is not None]
        if accuracies:  # Check if list is not empty
            accuracy_values.append(np.mean(accuracies))
            error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
            colors.append('#98df8a')  # Light green
        else:
            # Remove the category if no valid data
            categories.pop()
      # Create the plot with more height to accommodate x-axis labels
    plt.figure(figsize=(14, 10))
    
    # Check if we have any valid data points
    if not categories or not accuracy_values:
        plt.text(0.5, 0.5, "No valid data available for plotting", 
                ha='center', va='center', fontsize=BIGGER_SIZE, fontweight='bold')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        # Bar chart with error bars
        bars = plt.bar(categories, accuracy_values, color=colors, yerr=error_bars, 
                      capsize=10, edgecolor='black', linewidth=1.5, alpha=0.7)
        
        # Add exact values on top of bars
        for bar, value in zip(bars, accuracy_values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", 
                     ha='center', va='bottom', fontweight='bold', fontsize=MEDIUM_SIZE)
    
    plt.ylabel('Accuracy', fontsize=MEDIUM_SIZE, fontweight='bold')
    plt.ylim(0, max(accuracy_values) * 1.2)  # Add some space above the bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add more space at the bottom for x-axis labels
    plt.subplots_adjust(bottom=0.2)
    # Adjust x-tick positions and labels for better readability
    plt.tick_params(axis='x', which='major', labelsize=MEDIUM_SIZE)
    # Set font weight for x-axis tick labels
    for label in plt.gca().get_xticklabels():
        label.set_fontweight('bold')
      # Add count of runs for each category below the axis (only if we have categories)
    if categories:
        plt.annotate('Runs:', xy=(-0.15, -0.15), xycoords='axes fraction', fontsize=SMALL_SIZE, fontweight='bold')
        
        # Get counts for each category
        category_result_map = {
            "Static": static_results,
            "Adaptive\n(with ref)": adaptive_with_ref_results,
            "Adaptive\n(no ref)": adaptive_no_ref_results,
            "EMA\n(with ref)": ema_with_ref_results,
            "EMA\n(no ref)": ema_no_ref_results
        }
        
        # Count only results with valid accuracy values for each displayed category
        counts = []
        for category in categories:
            if category in category_result_map:
                valid_results = [r for r in category_result_map[category] if r.get('accuracy') is not None]
                counts.append(len(valid_results))
            else:
                counts.append(0)
                
        for i, (category, count) in enumerate(zip(categories, counts)):
            plt.annotate(f"n={count}", xy=(i, -0.1), xycoords=('data', 'axes fraction'), 
                         ha='center', fontsize=SMALL_SIZE)
      # Title
    if title:
        plt.title(title, fontweight='bold')
    else:
        plt.title('Accuracy Comparison by Agent Type', fontweight='bold')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.annotate(f"Generated: {timestamp}", xy=(1, -0.25), xycoords='axes fraction', 
                 ha='right', fontsize=8, alpha=0.7)
    
    # Use tight_layout but maintain the bottom adjustment we made
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
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
        use_initial_embeddings = result['metadata'].get('use_initial_embeddings')
        param_value = result['metadata'].get(parameter_name)
        
        # Create a more descriptive agent type label
        agent_type_label = f"{agent_type}"
        if use_initial_embeddings is not None:
            agent_type_label += " (with ref)" if use_initial_embeddings else " (no ref)"
            
        agent_types.add(agent_type_label)
        param_values.add(param_value)
        
        key = (agent_type_label, param_value)
        if key not in data:
            data[key] = []
        data[key].append(result['accuracy'])
    
    # Sort parameter values
    param_values = sorted(list(param_values))
    agent_types = sorted(list(agent_types))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Pick colors for each agent type
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_types)))
    for i, agent_type in enumerate(agent_types):
        agent_accuracies = []
        agent_errors = []
        for param_value in param_values:
            key = (agent_type, param_value)
            if key in data:
                # Filter out None values
                valid_values = [v for v in data[key] if v is not None]
                if valid_values:  # Check if we have any valid values
                    agent_accuracies.append(np.mean(valid_values))
                    agent_errors.append(np.std(valid_values) if len(valid_values) > 1 else 0)
                else:
                    agent_accuracies.append(None)  # No valid data for this combination
                    agent_errors.append(None)
            else:
                agent_accuracies.append(None)  # No data for this combination
                agent_errors.append(None)
        
        # Plot only if we have data for this agent type
        valid_indices = [i for i, x in enumerate(agent_accuracies) if x is not None]
        if valid_indices:
            valid_params = [param_values[i] for i in valid_indices]
            valid_accuracies = [agent_accuracies[i] for i in valid_indices]
            valid_errors = [agent_errors[i] for i in valid_indices]
              # Line plot with error bars
            plt.errorbar(valid_params, valid_accuracies, yerr=valid_errors, 
                        fmt='o-', label=agent_type, color=colors[i % len(colors)], 
                        capsize=5, linewidth=2, markersize=8, alpha=0.8)
            
            # Add values as text
            for j, (x, y, err) in enumerate(zip(valid_params, valid_accuracies, valid_errors)):
                plt.text(x, y + 0.02, f"{y:.2%}", ha='center', va='bottom', 
                        fontweight='bold', fontsize=9, color=colors[i % len(colors)])
                
                # Add sample size if we have runs
                key = (agent_type, x)  # Reconstruct the key for this specific point
                if key in data:
                    # Count valid data points
                    valid_points = [v for v in data[key] if v is not None]
                    if valid_points:
                        plt.text(x, y - 0.02, f"n={len(valid_points)}", ha='center', va='top', 
                                fontsize=7, alpha=0.7, color=colors[i % len(colors)])
    
    # Parameter labels
    param_display_name = {
        'confidence_threshold': 'Confidence Threshold',
        'ema_alpha': 'EMA Alpha',
        'motion_threshold': 'Motion Threshold',
        'bootstrapping_frames': 'Bootstrapping Frames'
    }.get(parameter_name, parameter_name)
    
    plt.xlabel(param_display_name, fontsize=MEDIUM_SIZE, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=MEDIUM_SIZE, fontweight='bold')
    
    # Title
    if title:
        plt.title(title, fontweight='bold')
    else:
        plt.title(f'Impact of {param_display_name} on Accuracy', fontweight='bold')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.annotate(f"Generated: {timestamp}", xy=(1, -0.1), xycoords='axes fraction', 
                 ha='right', fontsize=8, alpha=0.7)
    
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Ensure y-axis starts from 0
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy results for different agent types")
    parser.add_argument("--results_dir", help="Directory containing JSON result files")
    parser.add_argument("--output", help="Output directory for plots")
    parser.add_argument("--batch_name", default=None, help="Custom batch name for plots")
    parser.add_argument("--parameter", default=None, help="Parameter to analyze impact (e.g., confidence_threshold, ema_alpha)")
    args = parser.parse_args()

    # Load results
    results = load_results(args.results_dir)
    if not results:
        print("No result files found")
        return

    # Create output directory
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = args.results_dir

    # Determine batch name
    batch_name = args.batch_name
    if not batch_name:
        batch_name = os.path.basename(os.path.normpath(args.results_dir))
    
    # Overall accuracy plot
    output_path = os.path.join(output_dir, f"{batch_name}_accuracy_comparison.png")
    plot_overall_accuracy(results, output_path, title=f"Agent Accuracy Comparison - {batch_name}")
    
    # Parameter impact plots if requested
    if args.parameter:
        output_path = os.path.join(output_dir, f"{batch_name}_{args.parameter}_impact.png")
        plot_parameter_impact(results, args.parameter, output_path, 
                            title=f"Impact of {args.parameter} on Accuracy - {batch_name}")
    else:
        # Generate common parameter analysis plots
        for param in ['confidence_threshold', 'ema_alpha']:
            output_path = os.path.join(output_dir, f"{batch_name}_{param}_impact.png")
            plot_parameter_impact(results, param, output_path, 
                                title=f"Impact of {param} on Accuracy - {batch_name}")

if __name__ == "__main__":
    main()
