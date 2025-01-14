import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def prepare_data(results_dict: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Convert results dictionary into a pandas DataFrame for easier plotting"""
    data = []
    for agent_type, results in results_dict.items():
        for result in results:
            data.append({
                'Agent Type': agent_type.capitalize(),
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


def plot_performance_metrics(results_dict: Dict[str, List[Dict]], save_path: str = None):
    """Create a comprehensive performance comparison visualization"""
    set_style()
    df = prepare_data(results_dict)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Agent Performance Comparison', fontsize=16, y=0.95)
    
    # 1. Distribution of Time Taken
    sns.boxplot(data=df, x='Agent Type', y='Time Taken', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Time Taken')
    axes[0, 0].set_ylabel('Time Steps')
    
    # 2. Distribution of Moves Made
    sns.boxplot(data=df, x='Agent Type', y='Moves Made', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Moves Made')
    axes[0, 1].set_ylabel('Number of Moves')
    
    # 3. Success Rate
    success_rates = df.groupby('Agent Type')['Success'].mean() * 100
    sns.barplot(x=success_rates.index, y=success_rates.values, ax=axes[1, 0])
    axes[1, 0].set_title('Success Rate by Agent Type')
    axes[1, 0].set_ylabel('Success Rate (%)')
    
    # 4. Clean Percentage Distribution
    sns.violinplot(data=df, x='Agent Type', y='Clean Percentage', ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Clean Percentage')
    axes[1, 1].set_ylabel('Clean Percentage (%)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_efficiency_metrics(results_dict: Dict[str, List[Dict]], save_path: str = None):
    """Create visualizations focusing on efficiency metrics"""
    set_style()
    df = prepare_data(results_dict)
    
    # Calculate efficiency metrics
    df['Moves per Clean'] = df['Moves Made'] / (df['Clean Percentage'] / 100)
    df['Time per Clean'] = df['Time Taken'] / (df['Clean Percentage'] / 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Agent Efficiency Metrics', fontsize=16, y=0.95)
    
    # 1. Moves per Clean Percentage
    sns.boxplot(data=df, x='Agent Type', y='Moves per Clean', ax=axes[0, 0])
    axes[0, 0].set_title('Moves Required per Clean Percentage')
    axes[0, 0].set_ylabel('Moves/Clean%')
    
    # 2. Time per Clean Percentage
    sns.boxplot(data=df, x='Agent Type', y='Time per Clean', ax=axes[0, 1])
    axes[0, 1].set_title('Time Required per Clean Percentage')
    axes[0, 1].set_ylabel('Time/Clean%')
    
    # 3. Cleaning Rate Over Time
    clean_rate = df['Clean Percentage'] / df['Time Taken']
    sns.boxplot(data=df, x='Agent Type', y=clean_rate, ax=axes[1, 0])
    axes[1, 0].set_title('Cleaning Rate (% per Time Step)')
    axes[1, 0].set_ylabel('Clean% per Time Step')
    
    # 4. Movement Efficiency
    move_efficiency = df['Clean Percentage'] / df['Moves Made']
    sns.boxplot(data=df, x='Agent Type', y=move_efficiency, ax=axes[1, 1])
    axes[1, 1].set_title('Movement Efficiency (Clean% per Move)')
    axes[1, 1].set_ylabel('Clean% per Move')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_comparative_analysis(width: int, height: int, num_agents: int, 
                            dirty_percentage: float, max_time: int, 
                            num_trials: int, save_dir: str = None):
    """Run experiments and create comprehensive visualizations"""
    from .analysis import run_experiment
    
    # Run experiments for different agent types
    results = {}
    for agent_type in ['original', 'collaborative', 'mixed']:
        results[agent_type] = run_experiment(
            width, height, num_agents, dirty_percentage, 
            max_time, num_trials, agent_type
        )
    
    # Create and save performance metrics plot
    if save_dir:
        performance_path = f"{save_dir}/performance_metrics.png"
        efficiency_path = f"{save_dir}/efficiency_metrics.png"
    else:
        performance_path = efficiency_path = None
        
    fig1 = plot_performance_metrics(results, performance_path)
    fig2 = plot_efficiency_metrics(results, efficiency_path)
    
    return fig1, fig2


def create_performance_dashboard(base_results: List[Dict], agent_results: Dict, dirty_results: Dict):
    """
    Creates an interactive dashboard showing the performance metrics of the vacuum cleaner agents.
    
    Args:
        base_results: Results from the base configuration run
        agent_results: Results from varying number of agents
        dirty_results: Results from varying initial dirty percentage
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Time Distribution by Agent Type',
            'Moves Distribution by Agent Type',
            'Cleaning Efficiency Distribution',
            'Performance vs Number of Agents',
            'Movement Efficiency vs Number of Agents',
            'Cleaning Success vs Number of Agents',
            'Time vs Initial Dirt Level',
            'Movement vs Initial Dirt Level',
            'Cleaning Success vs Initial Dirt Level'
        ),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Distribution plots (top row)
    # Extract metrics by agent type
    agent_types = list(set(r.get('agent_type', 'mixed') for r in base_results))
    
    for idx, agent_type in enumerate(agent_types):
        type_results = [r for r in base_results if r.get('agent_type', 'mixed') == agent_type]
        
        # Time distribution
        fig.add_trace(
            go.Histogram(
                x=[r['time_taken'] for r in type_results],
                name=f'{agent_type.capitalize()}',
                opacity=0.75,
                marker_color=colors[idx],
                nbinsx=30
            ),
            row=1, col=1
        )
        
        # Moves distribution
        fig.add_trace(
            go.Histogram(
                x=[r['total_moves'] for r in type_results],
                name=f'{agent_type.capitalize()}',
                opacity=0.75,
                marker_color=colors[idx],
                nbinsx=30
            ),
            row=1, col=2
        )
        
        # Clean percentage distribution
        fig.add_trace(
            go.Histogram(
                x=[r['clean_percentage'] for r in type_results],
                name=f'{agent_type.capitalize()}',
                opacity=0.75,
                marker_color=colors[idx],
                nbinsx=30
            ),
            row=1, col=3
        )

    # Agent comparison plots (middle row)
    agents = sorted(agent_results.keys())
    
    # Performance metrics
    metrics = {
        'times': ('Average Time', 'Time Steps'),
        'moves': ('Total Moves', 'Number of Moves'),
        'clean': ('Cleaning Success', 'Clean Percentage (%)')
    }
    
    for col, (metric, (title, ylabel)) in enumerate(metrics.items(), start=1):
        avg_metric = [np.mean(agent_results[a][metric]) for a in agents]
        std_metric = [np.std(agent_results[a][metric]) for a in agents]
        
        fig.add_trace(
            go.Scatter(
                x=agents,
                y=avg_metric,
                error_y=dict(type='data', array=std_metric, visible=True),
                mode='lines+markers',
                name=title,
                line=dict(color=colors[col-1], width=2),
                marker=dict(size=8)
            ),
            row=2, col=col
        )

    # Dirty percentage comparison plots (bottom row)
    pcts = sorted(dirty_results.keys())
    
    for col, (metric, (title, ylabel)) in enumerate(metrics.items(), start=1):
        avg_metric = [np.mean(dirty_results[p][metric]) for p in pcts]
        std_metric = [np.std(dirty_results[p][metric]) for p in pcts]
        
        fig.add_trace(
            go.Scatter(
                x=pcts,
                y=avg_metric,
                error_y=dict(type='data', array=std_metric, visible=True),
                mode='lines+markers',
                name=title,
                line=dict(color=colors[col-1], width=2),
                marker=dict(size=8)
            ),
            row=3, col=col
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text='Vacuum Cleaner Agent Performance Analysis',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=1200,
        width=1600,
        template='plotly_white',
        font=dict(size=12)
    )

    # Update axes labels and styling
    for i in range(1, 10):
        row = (i-1) // 3 + 1
        col = (i-1) % 3 + 1
        
        # Update axes styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False,
            row=row,
            col=col
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False,
            row=row,
            col=col
        )

    return fig


def plot_agent_comparison(results: Dict[str, List[Dict]]):
    """
    Creates detailed comparison plots for different agent types.
    
    Args:
        results: Dictionary containing results for different agent types
    """
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Agent Type Performance Comparison', fontsize=16, y=0.95)
    
    # Prepare data
    agent_types = list(results.keys())
    metrics = {
        'time_taken': 'Time Steps',
        'total_moves': 'Total Moves',
        'clean_percentage': 'Clean Percentage (%)'
    }
    
    # Plot distributions
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        
        data = []
        labels = []
        for agent_type in agent_types:
            values = [r[metric] for r in results[agent_type]]
            data.append(values)
            labels.extend([agent_type.capitalize()] * len(values))
        
        sns.violinplot(data=data, ax=ax)
        ax.set_xticklabels([t.capitalize() for t in agent_types])
        ax.set_ylabel(label)
        ax.set_title(f'{label} Distribution by Agent Type')
        
    # Plot success rate comparison
    ax = axes[1, 1]
    success_rates = []
    for agent_type in agent_types:
        success = sum(1 for r in results[agent_type] if r['clean_percentage'] == 100)
        rate = success / len(results[agent_type]) * 100
        success_rates.append(rate)
    
    bars = ax.bar(range(len(agent_types)), success_rates)
    ax.set_xticks(range(len(agent_types)))
    ax.set_xticklabels([t.capitalize() for t in agent_types])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Cleaning Success Rate by Agent Type')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def create_distribution_plots(fig, times, moves, clean_percentages):
    """Add distribution plots to the figure for time, moves, and clean percentages.
    
    Args:
        fig: Plotly figure object
        times: List of completion times
        moves: List of total moves
        clean_percentages: List of clean percentages
    """
    metrics = [(times, 'Time'), (moves, 'Moves'), (clean_percentages, 'Clean %')]
    
    for idx, (data, name) in enumerate(metrics, 1):
        fig.add_trace(
            go.Histogram(x=data, name=f'{name} Distribution', nbinsx=20),
            row=1, col=idx
        )
        fig.add_vline(x=np.mean(data), line_dash="dash", line_color="red",
                      annotation_text="Mean", row=1, col=idx)


def create_agent_comparison_plots(fig, agent_results):
    """Add agent comparison plots to the figure.
    
    Args:
        fig: Plotly figure object
        agent_results: Dictionary containing results for different agent counts
    """
    agents = sorted(agent_results.keys())
    metrics = ['times', 'moves', 'clean']
    labels = ['Time', 'Moves', 'Clean %']
    
    for idx, (metric, label) in enumerate(zip(metrics, labels), 1):
        avgs = [np.mean(agent_results[a][metric]) for a in agents]
        stds = [np.std(agent_results[a][metric]) for a in agents]
        fig.add_trace(
            go.Scatter(x=agents, y=avgs, mode='markers', 
                      name=f'Average {label}',
                      error_y=dict(type='data', array=stds, visible=True)),
            row=2, col=idx
        )


def create_dirty_comparison_plots(fig, dirty_results):
    """Add dirty percentage comparison plots to the figure.
    
    Args:
        fig: Plotly figure object
        dirty_results: Dictionary containing results for different initial dirty percentages
    """
    pcts = sorted(dirty_results.keys())
    metrics = ['times', 'moves', 'clean']
    labels = ['Time', 'Moves', 'Clean %']
    
    for idx, (metric, label) in enumerate(zip(metrics, labels), 1):
        avgs = [np.mean(dirty_results[p][metric]) for p in pcts]
        stds = [np.std(dirty_results[p][metric]) for p in pcts]
        fig.add_trace(
            go.Scatter(x=pcts, y=avgs, mode='markers',
                      name=f'Average {label}',
                      error_y=dict(type='data', array=stds, visible=True)),
            row=3, col=idx
        )


def plot_all_metrics(base_results, agent_results, dirty_results):
    """Create a comprehensive visualization of all metrics from the simulation.
    
    Args:
        base_results: List of results from base simulation runs
        agent_results: Dictionary of results with varying agent counts
        dirty_results: Dictionary of results with varying initial dirty percentages
    
    Returns:
        Plotly figure object containing all plots
    """
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Distribution of Time Taken', 'Distribution of Moves Made', 'Distribution of Clean Percentage',
            'Average Time vs Number of Agents', 'Average Moves vs Number of Agents', 'Average Clean % vs Number of Agents',
            'Average Time vs Initial Dirty %', 'Average Moves vs Initial Dirty %', 'Average Clean % vs Initial Dirty %'
        )
    )
    
    # Extract base metrics
    times = [r['time_taken'] for r in base_results]
    moves = [r['total_moves'] for r in base_results]
    clean_percentages = [r['clean_percentage'] for r in base_results]

    # Create all plots
    create_distribution_plots(fig, times, moves, clean_percentages)
    create_agent_comparison_plots(fig, agent_results)
    create_dirty_comparison_plots(fig, dirty_results)

    # Update layout
    fig.update_layout(height=900, width=1200, showlegend=False)
    return fig


def run_analysis_visualization(width: int, height: int, max_time: int, num_trials: int):
    """Run a comprehensive analysis and create visualizations.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        max_time: Maximum simulation time
        num_trials: Number of trials to run
    """
    # Test different numbers of agents
    agent_counts = [2, 4, 6, 8]
    dirty_percentage = 0.5
    agent_results = {}
    
    for num_agents in agent_counts:
        results = run_experiment(width, height, num_agents, dirty_percentage, max_time, num_trials)
        agent_results[num_agents] = analyze_results(results)
    
    # Test different initial dirty percentages
    num_agents = 4
    dirty_percentages = [0.3, 0.5, 0.7, 0.9]
    dirty_results = {}
    
    for dirty_pct in dirty_percentages:
        results = run_experiment(width, height, num_agents, dirty_pct, max_time, num_trials)
        dirty_results[dirty_pct] = analyze_results(results)
    
    # Run base configuration multiple times
    base_results = run_experiment(width, height, num_agents, dirty_percentage, max_time, num_trials)
    
    # Create and display all visualizations
    plot_all_metrics(base_results, agent_results, dirty_results)
    plot_performance_metrics(compare_agent_types(width, height, max_time, num_trials))
    plot_efficiency_metrics(compare_agent_types(width, height, max_time, num_trials))
    plot_agent_comparison(compare_agent_types(width, height, max_time, num_trials))
