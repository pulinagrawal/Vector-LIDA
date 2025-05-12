#!/usr/bin/env python
# filepath: c:\Users\Nathan\CVResearch\Vector-LIDA\film_agent\plot_results_no_selflearning.py
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
        accuracies = [r['accuracy'] for r in static_results]
        accuracy_values.append(np.mean(accuracies))
        error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
        colors.append('#7f7f7f')  # Gray
    
    # Add adaptive with reference embeddings
    if adaptive_with_ref_results:
        categories.append("Adaptive\n(with ref)")
        accuracies = [r['accuracy'] for r in adaptive_with_ref_results]
        accuracy_values.append(np.mean(accuracies))
        error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
        colors.append('#1f77b4')  # Blue
    
    # Add adaptive without reference embeddings
    if adaptive_no_ref_results:
        categories.append("Adaptive\n(no ref)")
        accuracies = [r['accuracy'] for r in adaptive_no_ref_results]
        accuracy_values.append(np.mean(accuracies))
        error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
        colors.append('#aec7e8')  # Light blue
    
    # Add EMA with reference embeddings
    if ema_with_ref_results:
        categories.append("EMA\n(with ref)")
        accuracies = [r['accuracy'] for r in ema_with_ref_results]
        accuracy_values.append(np.mean(accuracies))
        error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
        colors.append('#2ca02c')  # Green
      # Add EMA without reference embeddings
    if ema_no_ref_results:
        categories.append("EMA\n(no ref)")
        accuracies = [r['accuracy'] for r in ema_no_ref_results]
        accuracy_values.append(np.mean(accuracies))
        error_bars.append(np.std(accuracies) if len(accuracies) > 1 else 0)
        colors.append('#98df8a')  # Light green
    
    # Create the plot with more height to accommodate x-axis labels
    plt.figure(figsize=(14, 10))
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
    
    # Add count of runs for each category below the axis
    plt.annotate('Runs:', xy=(-0.15, -0.15), xycoords='axes fraction', fontsize=SMALL_SIZE, fontweight='bold')
    counts = [
        len(static_results), 
        len(adaptive_with_ref_results), 
        len(adaptive_no_ref_results), 
        len(ema_with_ref_results), 
        len(ema_no_ref_results)
    ]
    
    # Only include categories with data
    counts = [count for i, count in enumerate(counts) if i < len(categories)]
    
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
    """Plot the impact of a specific parameter on accuracy, separating EMA and Adaptive agents"""
    # Define which parameters are valid for which agent types
    parameter_agent_mapping = {
        'confidence_threshold': ['adaptive', 'ema'],
        'ema_alpha': ['ema'],  # EMA alpha only applies to EMA agents
        'motion_threshold': ['adaptive', 'ema'],
        'bootstrapping_frames': ['adaptive', 'ema']
    }
    
    # Filter results that have the specified parameter
    filtered_results = [r for r in results if parameter_name in r['metadata'] and r['metadata'][parameter_name] is not None]
    
    if not filtered_results:
        print(f"No results found with parameter '{parameter_name}'")
        return None

    # Get agent types that should use this parameter
    valid_agent_types = parameter_agent_mapping.get(parameter_name, ['adaptive', 'ema'])
    
    # Separate results for EMA and Adaptive agents
    ema_results = [r for r in filtered_results if r['metadata'].get('agent_type') == 'ema']
    adaptive_results = [r for r in filtered_results if r['metadata'].get('agent_type') == 'adaptive']

    def plot_agent_type(agent_results, agent_type, output_path_suffix):
        # Skip plotting if this agent type doesn't use this parameter
        if agent_type.lower() not in valid_agent_types:
            print(f"Parameter '{parameter_name}' is not used by {agent_type} agents - skipping plot")
            return None
        
        if not agent_results:
            print(f"No results found for {agent_type} agents with parameter '{parameter_name}'")
            return None

        # Group by parameter value and embedding usage
        param_values = set()
        data = {}

        for result in agent_results:
            use_initial_embeddings = result['metadata'].get('use_initial_embeddings')
            param_value = result['metadata'].get(parameter_name)

            # Create a descriptive key combining reference usage
            key = f"{'with_ref' if use_initial_embeddings else 'no_ref'}"

            param_values.add(param_value)

            if key not in data:
                data[key] = {}
            
            if param_value not in data[key]:
                data[key][param_value] = []
                
            data[key][param_value].append(result['accuracy'])

        # Sort parameter values
        param_values = sorted(list(param_values))

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Pick colors for each embedding usage
        colors = {'with_ref': '#1f77b4', 'no_ref': '#aec7e8'}
        
        # Track min/max y values for dynamic y-axis scaling
        min_y_value = float('inf')
        max_y_value = float('-inf')
        
        for key, color in colors.items():
            if key in data:
                x_values = []
                y_values = []
                std_values = []

                for param_value in param_values:
                    if param_value in data[key]:
                        values = data[key][param_value]
                        if values:  # Check if we have any values
                            x_values.append(param_value)
                            mean_value = np.mean(values)
                            y_values.append(mean_value)
                            std_val = np.std(values) if len(values) > 1 else 0
                            std_values.append(std_val)
                            
                            # Update min/max for y-axis scaling
                            min_y_value = min(min_y_value, mean_value - std_val)
                            max_y_value = max(max_y_value, mean_value + std_val)

                if x_values:  # Only plot if we have data points
                    plt.errorbar(x_values, y_values, yerr=std_values, 
                                fmt='o-', label=f"{key}", color=color, 
                                capsize=5, linewidth=2, markersize=8, alpha=0.8)

                    # Add values as text
                    for x, y, err in zip(x_values, y_values, std_values):
                        plt.text(x, y + 0.02, f"{y:.2%}", ha='center', va='bottom', 
                                fontweight='bold', fontsize=9, color=color)

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
            plt.title(f"{title} - {agent_type}", fontweight='bold')
        else:
            plt.title(f'Impact of {param_display_name} on Accuracy - {agent_type}', fontweight='bold')

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.annotate(f"Generated: {timestamp}", xy=(1, -0.1), xycoords='axes fraction', 
                     ha='right', fontsize=8, alpha=0.7)

        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        # Dynamic y-axis scaling with padding
        if min_y_value != float('inf') and max_y_value != float('-inf'):
            y_padding = (max_y_value - min_y_value) * 0.1  # 10% padding
            plt.ylim(max(0, min_y_value - y_padding), max_y_value + y_padding)

        plt.tight_layout()

        # Save or show the plot
        if output_path:
            agent_output_path = output_path.replace(".png", f"_{agent_type.lower()}.png")
            plt.savefig(agent_output_path, bbox_inches='tight', dpi=300)
            print(f"Saved plot to {agent_output_path}")
        else:
            plt.show()

        return plt.gcf()

    plot_results = []
    
    # Plot for EMA agents if parameter applies to them
    if 'ema' in valid_agent_types:
        ema_fig = plot_agent_type(ema_results, "EMA", output_path)
        if ema_fig:
            plot_results.append(ema_fig)
        else:
            print(f"INFO: No plot was created for EMA agents with parameter '{parameter_name}'")

    # Plot for Adaptive agents if parameter applies to them
    if 'adaptive' in valid_agent_types:
        adaptive_fig = plot_agent_type(adaptive_results, "Adaptive", output_path)
        if adaptive_fig:
            plot_results.append(adaptive_fig)
        else:
            print(f"INFO: No plot was created for Adaptive agents with parameter '{parameter_name}'")
            
    if not plot_results:
        print(f"WARNING: No plots were created for parameter '{parameter_name}'")
            
    return plot_results

def plot_confidence_comparison(results, output_path=None, title=None):
    """Generate a specialized plot comparing confidence thresholds for different agent types"""
    # Filter results that have confidence_threshold parameter
    filtered_results = [r for r in results if 'confidence_threshold' in r['metadata'] 
                        and r['metadata']['confidence_threshold'] is not None]
    
    if not filtered_results:
        print("No results found with confidence threshold parameter")
        return None
        
    # Separate by agent type
    ema_results = [r for r in filtered_results if r['metadata'].get('agent_type') == 'ema']
    adaptive_results = [r for r in filtered_results if r['metadata'].get('agent_type') == 'adaptive']
    
    plt.figure(figsize=(14, 10))
    
    # Track min/max y values for dynamic y-axis scaling
    all_y_values = []
    
    # Plot for each agent type
    for agent_results, agent_type, color_base in [
        (ema_results, 'EMA', '#2ca02c'), 
        (adaptive_results, 'Adaptive', '#1f77b4')
    ]:
        # Group by confidence threshold and embedding usage
        conf_values = {}
        
        for result in agent_results:
            use_initial_embeddings = result['metadata'].get('use_initial_embeddings')
            conf = result['metadata'].get('confidence_threshold')
            
            key = f"{agent_type} ({'with ref' if use_initial_embeddings else 'no ref'})"
            
            if key not in conf_values:
                conf_values[key] = {}
            
            if conf not in conf_values[key]:
                conf_values[key][conf] = []
                
            conf_values[key][conf].append(result['accuracy'])
                
        # Plot data
        for i, (key, values) in enumerate(conf_values.items()):
            color = color_base if 'with ref' in key else f"#{int(color_base[1:3], 16):02x}{int(color_base[3:5], 16):02x}{min(int(color_base[5:7], 16) + 90, 255):02x}"
            
            x_values = []
            y_values = []
            err_values = []
            
            for conf, accuracies in sorted(values.items()):
                x_values.append(conf)
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies) if len(accuracies) > 1 else 0
                y_values.append(mean_accuracy)
                err_values.append(std_accuracy)
                
                # Store for dynamic y-axis scaling
                all_y_values.append(mean_accuracy - std_accuracy)
                all_y_values.append(mean_accuracy + std_accuracy)
            
            if x_values:
                plt.errorbar(x_values, y_values, yerr=err_values,
                          fmt='o-', label=key, color=color,
                          linewidth=2, markersize=8, capsize=5)
                
                # Add text values
                for x, y in zip(x_values, y_values):
                    plt.text(x, y + 0.01, f"{y:.2%}", ha='center', va='bottom',
                            fontweight='bold', fontsize=9, color=color)
    
    plt.xlabel('Confidence Threshold', fontsize=MEDIUM_SIZE, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=MEDIUM_SIZE, fontweight='bold')
    
    # Title
    if title:
        plt.title(title, fontweight='bold')
    else:
        plt.title('Impact of Confidence Threshold on Accuracy', fontweight='bold')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.annotate(f"Generated: {timestamp}", xy=(1, -0.1), xycoords='axes fraction', 
                ha='right', fontsize=8, alpha=0.7)
    
    plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # More compact y-axis scaling with reduced padding
    if all_y_values:
        min_y = min(all_y_values)
        max_y = max(all_y_values)
        y_range = max_y - min_y
        
        # Use a smaller range by setting a reasonable bottom limit and reducing top padding
        # Find a reasonable minimum that's close to the lowest data point but still looks good
        bottom_limit = max(0.5, min_y - y_range * 0.02)  # Never go below 0.5 for accuracy plots
        # Add just enough padding at the top for annotations but keep it compact
        plt.ylim(bottom_limit, max_y + y_range * 0.08)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy results for different agent types")
    parser.add_argument("results_dir", help="Directory containing JSON result files")
    parser.add_argument("--output-dir", default="plots", help="Directory to save output plots")
    parser.add_argument("--overall", action="store_true", help="Generate overall accuracy comparison plot")
    parser.add_argument("--confidence", action="store_true", help="Generate confidence threshold comparison plot")
    parser.add_argument("--ema-alpha", action="store_true", help="Generate EMA alpha parameter impact plot")
    parser.add_argument("--motion", action="store_true", help="Generate motion threshold parameter impact plot")
    parser.add_argument("--bootstrap", action="store_true", help="Generate bootstrapping frames parameter impact plot")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--batch-name", default="batch", help="Name for the batch of results (used in filenames)")
    parser.add_argument("--confidence-only", action="store_true", help="Generate only the confidence threshold comparison plot for adaptive and EMA agents")
    parser.add_argument("--parameter", help="Generate a parameter impact plot for the specified parameter")
    args = parser.parse_args()

    # Load results
    results = load_results(args.results_dir)
    if not results:
        print("No result files found")
        return

    # Create output directory
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = args.results_dir

    # Determine batch name
    batch_name = args.batch_name
    if not batch_name:
        batch_name = os.path.basename(os.path.normpath(args.results_dir))
    
    # Check if we should only generate the confidence comparison plot
    if args.confidence_only:
        output_path = os.path.join(output_dir, f"{batch_name}_confidence_comparison.png")
        plot_confidence_comparison(results, output_path, 
                                  title=f"Confidence Threshold Impact - {batch_name}")
        return
    
    # Generate plots based on arguments
    if args.overall or args.all:
        output_path = os.path.join(output_dir, f"{batch_name}_accuracy_comparison.png")
        plot_overall_accuracy(results, output_path, 
                            title=f"Agent Accuracy Comparison - {batch_name}")
    
    # Parameter impact plots
    if args.parameter:
        output_path = os.path.join(output_dir, f"{batch_name}_{args.parameter}_impact.png")
        plot_parameter_impact(results, args.parameter, output_path, 
                            title=f"Impact of {args.parameter} on Accuracy - {batch_name}")
    
    # Generate specific parameter plots if requested
    if args.confidence or args.all:
        output_path = os.path.join(output_dir, f"{batch_name}_confidence_impact.png")
        plot_parameter_impact(results, 'confidence_threshold', output_path, 
                             title=f"Impact of Confidence Threshold - {batch_name}")
        
        # Also generate the specialized confidence comparison plot
        output_path = os.path.join(output_dir, f"{batch_name}_confidence_comparison.png")
        plot_confidence_comparison(results, output_path, 
                                 title=f"Confidence Threshold Impact - {batch_name}")
    
    if args.ema_alpha or args.all:
        output_path = os.path.join(output_dir, f"{batch_name}_ema_alpha_impact.png")
        plot_parameter_impact(results, 'ema_alpha', output_path, 
                             title=f"Impact of EMA Alpha - {batch_name}")
    
    if args.motion or args.all:
        output_path = os.path.join(output_dir, f"{batch_name}_motion_threshold_impact.png")
        plot_parameter_impact(results, 'motion_threshold', output_path, 
                             title=f"Impact of Motion Threshold - {batch_name}")
    
    if args.bootstrap or args.all:
        output_path = os.path.join(output_dir, f"{batch_name}_bootstrapping_frames_impact.png")
        plot_parameter_impact(results, 'bootstrapping_frames', output_path, 
                             title=f"Impact of Bootstrapping Frames - {batch_name}")
                             
    # If no specific plots were requested, generate default plots
    if not any([args.overall, args.confidence, args.ema_alpha, args.motion, 
                args.bootstrap, args.all, args.parameter]):
        # Overall accuracy plot
        output_path = os.path.join(output_dir, f"{batch_name}_accuracy_comparison.png")
        plot_overall_accuracy(results, output_path, 
                             title=f"Agent Accuracy Comparison - {batch_name}")
        
        # Generate common parameter analysis plots
        for param in ['confidence_threshold', 'ema_alpha']:
            output_path = os.path.join(output_dir, f"{batch_name}_{param}_impact.png")
            plot_parameter_impact(results, param, output_path, 
                                 title=f"Impact of {param} on Accuracy - {batch_name}")

if __name__ == "__main__":
    main()
