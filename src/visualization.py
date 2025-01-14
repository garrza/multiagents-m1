import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List


def prepare_data(results_dict: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Convert results dictionary into a pandas DataFrame for easier plotting"""
    data = []
    for agent_type, results in results_dict.items():
        # Format agent type name for better readability
        formatted_type = agent_type.replace('_', ' ').title()
        for result in results:
            data.append({
                'Agent Type': formatted_type,
                'Time Taken': result['time_taken'],
                'Moves Made': result['total_moves'],
                'Clean Percentage': result['clean_percentage'],
                'Success': result['clean_percentage'] == 100
            })
    return pd.DataFrame(data)


def set_style():
    """Set the style for all plots"""
    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn-v0_8-darkgrid")
    # Set color palette for better distinction between agent types
    sns.set_palette("husl")


def plot_performance_metrics(results_dict: Dict[str, List[Dict]], save_path: str = None):
    """Create a comprehensive performance comparison visualization"""
    set_style()
    df = prepare_data(results_dict)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Agent Performance Comparison', fontsize=16, y=0.95)
    
    # Custom color palette for agent types
    agent_types = df['Agent Type'].unique()
    colors = sns.color_palette("husl", n_colors=len(agent_types))
    agent_colors = dict(zip(agent_types, colors))
    
    # 1. Distribution of Time Taken
    sns.boxplot(data=df, x='Agent Type', y='Time Taken', ax=axes[0, 0], palette=agent_colors)
    axes[0, 0].set_title('Distribution of Time Taken', fontsize=12, pad=10)
    axes[0, 0].set_ylabel('Time Steps', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Distribution of Moves Made
    sns.boxplot(data=df, x='Agent Type', y='Moves Made', ax=axes[0, 1], palette=agent_colors)
    axes[0, 1].set_title('Distribution of Moves Made', fontsize=12, pad=10)
    axes[0, 1].set_ylabel('Number of Moves', fontsize=10)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Success Rate
    success_rates = df.groupby('Agent Type')['Success'].mean() * 100
    sns.barplot(x=success_rates.index, y=success_rates.values, ax=axes[1, 0], palette=agent_colors)
    axes[1, 0].set_title('Success Rate by Agent Type', fontsize=12, pad=10)
    axes[1, 0].set_ylabel('Success Rate (%)', fontsize=10)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Clean Percentage Distribution
    sns.violinplot(data=df, x='Agent Type', y='Clean Percentage', ax=axes[1, 1], palette=agent_colors)
    axes[1, 1].set_title('Distribution of Clean Percentage', fontsize=12, pad=10)
    axes[1, 1].set_ylabel('Clean Percentage (%)', fontsize=10)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_efficiency_metrics(results_dict: Dict[str, List[Dict]], save_path: str = None):
    """Create visualizations focusing on efficiency metrics"""
    set_style()
    df = prepare_data(results_dict)
    
    # Calculate efficiency metrics
    df['Moves per Clean'] = df['Moves Made'] / (df['Clean Percentage'] / 100)
    df['Time per Clean'] = df['Time Taken'] / (df['Clean Percentage'] / 100)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Agent Efficiency Metrics', fontsize=16, y=0.95)
    
    # Custom color palette for agent types
    agent_types = df['Agent Type'].unique()
    colors = sns.color_palette("husl", n_colors=len(agent_types))
    agent_colors = dict(zip(agent_types, colors))
    
    # 1. Moves per Clean Percentage
    sns.boxplot(data=df, x='Agent Type', y='Moves per Clean', ax=axes[0, 0], palette=agent_colors)
    axes[0, 0].set_title('Moves Required per Clean Percentage', fontsize=12, pad=10)
    axes[0, 0].set_ylabel('Moves/Clean%', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Time per Clean Percentage
    sns.boxplot(data=df, x='Agent Type', y='Time per Clean', ax=axes[0, 1], palette=agent_colors)
    axes[0, 1].set_title('Time Required per Clean Percentage', fontsize=12, pad=10)
    axes[0, 1].set_ylabel('Time/Clean%', fontsize=10)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Cleaning Rate Over Time
    clean_rate = df['Clean Percentage'] / df['Time Taken']
    df['Clean Rate'] = clean_rate
    sns.boxplot(data=df, x='Agent Type', y='Clean Rate', ax=axes[1, 0], palette=agent_colors)
    axes[1, 0].set_title('Cleaning Rate (% per Time Step)', fontsize=12, pad=10)
    axes[1, 0].set_ylabel('Clean% per Time Step', fontsize=10)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Movement Efficiency
    move_efficiency = df['Clean Percentage'] / df['Moves Made']
    df['Move Efficiency'] = move_efficiency
    sns.boxplot(data=df, x='Agent Type', y='Move Efficiency', ax=axes[1, 1], palette=agent_colors)
    axes[1, 1].set_title('Movement Efficiency (Clean% per Move)', fontsize=12, pad=10)
    axes[1, 1].set_ylabel('Clean% per Move', fontsize=10)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_agent_comparison(results: Dict[str, List[Dict]]):
    """
    Creates detailed comparison plots for different agent types.
    
    Args:
        results: Dictionary containing results for different agent types
    """
    set_style()
    df = prepare_data(results)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Agent Type Comparison Analysis', fontsize=16, y=0.95)
    
    # Custom color palette
    agent_types = df['Agent Type'].unique()
    colors = sns.color_palette("husl", n_colors=len(agent_types))
    agent_colors = dict(zip(agent_types, colors))
    
    # 1. Time vs Clean Percentage
    sns.scatterplot(data=df, x='Time Taken', y='Clean Percentage', 
                   hue='Agent Type', ax=axes[0, 0], palette=agent_colors)
    axes[0, 0].set_title('Time vs Clean Percentage', fontsize=12)
    
    # 2. Moves vs Clean Percentage
    sns.scatterplot(data=df, x='Moves Made', y='Clean Percentage', 
                   hue='Agent Type', ax=axes[0, 1], palette=agent_colors)
    axes[0, 1].set_title('Moves vs Clean Percentage', fontsize=12)
    
    # 3. Efficiency Metrics
    df['Efficiency'] = df['Clean Percentage'] / df['Time Taken']
    sns.boxplot(data=df, x='Agent Type', y='Efficiency', 
                ax=axes[1, 0], palette=agent_colors)
    axes[1, 0].set_title('Cleaning Efficiency (Clean% / Time)', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Movement Efficiency
    df['Move Efficiency'] = df['Clean Percentage'] / df['Moves Made']
    sns.boxplot(data=df, x='Agent Type', y='Move Efficiency', 
                ax=axes[1, 1], palette=agent_colors)
    axes[1, 1].set_title('Movement Efficiency (Clean% / Moves)', fontsize=12)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_comparative_summary(results_dict: Dict[str, List[Dict]]):
    """
    Create a summary visualization of the comparative analysis with helpful labels.
    
    Args:
        results_dict: Dictionary containing results for different agent types
    """
    set_style()
    df = prepare_data(results_dict)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle('Comparative Analysis Summary', fontsize=16, y=0.95)
    
    # Custom color palette
    agent_types = df['Agent Type'].unique()
    colors = sns.color_palette("husl", n_colors=len(agent_types))
    agent_colors = dict(zip(agent_types, colors))
    
    # 1. Time Distribution
    sns.violinplot(data=df, x='Agent Type', y='Time Taken', 
                  ax=axes[0, 0], palette=agent_colors)
    axes[0, 0].set_title('Time Distribution', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Moves Distribution
    sns.violinplot(data=df, x='Agent Type', y='Moves Made', 
                  ax=axes[0, 1], palette=agent_colors)
    axes[0, 1].set_title('Moves Distribution', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Clean Percentage vs Time
    sns.scatterplot(data=df, x='Time Taken', y='Clean Percentage', 
                   hue='Agent Type', ax=axes[1, 0], palette=agent_colors)
    axes[1, 0].set_title('Clean Percentage vs Time', fontsize=12)
    
    # 4. Clean Percentage vs Moves
    sns.scatterplot(data=df, x='Moves Made', y='Clean Percentage', 
                   hue='Agent Type', ax=axes[1, 1], palette=agent_colors)
    axes[1, 1].set_title('Clean Percentage vs Moves', fontsize=12)
    
    # 5. Success Rate
    success_rates = df.groupby('Agent Type')['Success'].mean() * 100
    sns.barplot(x=success_rates.index, y=success_rates.values, 
                ax=axes[2, 0], palette=agent_colors)
    axes[2, 0].set_title('Success Rate by Agent Type', fontsize=12)
    axes[2, 0].set_ylabel('Success Rate (%)')
    axes[2, 0].tick_params(axis='x', rotation=45)
    
    # 6. Efficiency Score
    df['Efficiency'] = (df['Clean Percentage'] / df['Time Taken']) * (df['Clean Percentage'] / df['Moves Made'])
    sns.boxplot(data=df, x='Agent Type', y='Efficiency', 
                ax=axes[2, 1], palette=agent_colors)
    axes[2, 1].set_title('Overall Efficiency Score', fontsize=12)
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def run_analysis_visualization(base_results: List[Dict], agent_results: Dict, dirty_results: Dict):
    """
    Run a comprehensive analysis and create visualizations.
    
    Args:
        base_results: Results from the base configuration run
        agent_results: Dictionary of results with varying number of agents
        dirty_results: Dictionary of results with varying initial dirty percentage
    """
    set_style()
    
    # Create figure with subplots for base results
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 15))
    fig1.suptitle('Base Configuration Analysis', fontsize=16, y=0.95)
    
    # Extract metrics from base results
    times = [r['time_taken'] for r in base_results]
    moves = [r['total_moves'] for r in base_results]
    clean_pcts = [r['clean_percentage'] for r in base_results]
    
    # Plot distributions for base results
    sns.histplot(times, ax=axes1[0, 0], kde=True)
    axes1[0, 0].set_title('Distribution of Completion Times')
    axes1[0, 0].set_xlabel('Time Steps')
    
    sns.histplot(moves, ax=axes1[0, 1], kde=True)
    axes1[0, 1].set_title('Distribution of Total Moves')
    axes1[0, 1].set_xlabel('Number of Moves')
    
    sns.histplot(clean_pcts, ax=axes1[1, 0], kde=True)
    axes1[1, 0].set_title('Distribution of Clean Percentages')
    axes1[1, 0].set_xlabel('Clean Percentage (%)')
    
    # Correlation plot
    df_base = pd.DataFrame({
        'Time': times,
        'Moves': moves,
        'Clean%': clean_pcts
    })
    sns.scatterplot(data=df_base, x='Time', y='Clean%', ax=axes1[1, 1])
    axes1[1, 1].set_title('Time vs Clean Percentage')
    
    plt.tight_layout()
    
    # Create figure for agent comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 15))
    fig2.suptitle('Agent Count Analysis', fontsize=16, y=0.95)
    
    # Prepare data for agent comparison
    agent_counts = sorted(agent_results.keys())
    avg_times = [np.mean(agent_results[a]['times']) for a in agent_counts]
    avg_moves = [np.mean(agent_results[a]['moves']) for a in agent_counts]
    avg_clean = [np.mean(agent_results[a]['clean']) for a in agent_counts]
    
    # Plot agent comparison metrics
    sns.lineplot(x=agent_counts, y=avg_times, ax=axes2[0, 0], marker='o')
    axes2[0, 0].set_title('Average Time vs Number of Agents')
    axes2[0, 0].set_xlabel('Number of Agents')
    axes2[0, 0].set_ylabel('Average Time Steps')
    
    sns.lineplot(x=agent_counts, y=avg_moves, ax=axes2[0, 1], marker='o')
    axes2[0, 1].set_title('Average Moves vs Number of Agents')
    axes2[0, 1].set_xlabel('Number of Agents')
    axes2[0, 1].set_ylabel('Average Moves')
    
    sns.lineplot(x=agent_counts, y=avg_clean, ax=axes2[1, 0], marker='o')
    axes2[1, 0].set_title('Average Clean Percentage vs Number of Agents')
    axes2[1, 0].set_xlabel('Number of Agents')
    axes2[1, 0].set_ylabel('Average Clean Percentage')
    
    # Create figure for dirty percentage comparison
    dirty_pcts = sorted(dirty_results.keys())
    avg_times_dirty = [np.mean(dirty_results[p]['times']) for p in dirty_pcts]
    avg_moves_dirty = [np.mean(dirty_results[p]['moves']) for p in dirty_pcts]
    avg_clean_dirty = [np.mean(dirty_results[p]['clean']) for p in dirty_pcts]
    
    sns.lineplot(x=dirty_pcts, y=avg_times_dirty, ax=axes2[1, 1], marker='o')
    axes2[1, 1].set_title('Performance vs Initial Dirty Percentage')
    axes2[1, 1].set_xlabel('Initial Dirty Percentage')
    axes2[1, 1].set_ylabel('Average Time Steps')
    
    plt.tight_layout()
    
    # Display all figures
    plt.show()
