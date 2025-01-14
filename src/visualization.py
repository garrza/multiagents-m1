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
    # Prepare data
    df = prepare_data(results)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Vacuum Cleaner Agent Performance Analysis', fontsize=14, pad=20)
    
    # Color palette
    colors = sns.color_palette("husl", n_colors=len(df['Agent Type'].unique()))
    
    # Time Distribution Plot
    sns.violinplot(data=df, x='Agent Type', y='Time Taken', ax=ax1, 
                  inner='box', palette=colors)
    ax1.set_title('Time Distribution by Agent Type', fontsize=12, pad=10)
    ax1.set_ylabel('Time Steps', fontsize=10)
    ax1.set_xlabel('Agent Type', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add grid for better readability
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    
    # Moves Distribution Plot
    sns.violinplot(data=df, x='Agent Type', y='Moves Made', ax=ax2,
                  inner='box', palette=colors)
    ax2.set_title('Moves Distribution by Agent Type', fontsize=12, pad=10)
    ax2.set_ylabel('Number of Moves', fontsize=10)
    ax2.set_xlabel('Agent Type', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add grid for better readability
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


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


def plot_comparative_summary(results_dict: Dict[str, List[Dict]]):
    """Create a summary visualization of the comparative analysis with helpful labels.
    
    Args:
        results_dict: Dictionary containing results for different agent types
    """
    # Calculate averages for each configuration
    summary_data = {}
    for agent_type, results in results_dict.items():
        avg_time = np.mean([r['time_taken'] for r in results])
        avg_moves = np.mean([r['total_moves'] for r in results])
        avg_clean = np.mean([r['clean_percentage'] for r in results])
        success_rate = np.mean([1 if r['clean_percentage'] == 100 else 0 for r in results])
        
        summary_data[agent_type] = {
            'Time Steps': avg_time,
            'Total Moves': avg_moves,
            'Clean %': avg_clean,
            'Success Rate': success_rate
        }
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Average Time Steps (Lower is Better)',
            'Average Total Moves (Lower is Better)',
            'Average Clean Percentage (Higher is Better)',
            'Success Rate (Higher is Better)'
        )
    )
    
    # Custom color palette
    colors = {
        'Mixed': '#2ecc71',      # Green
        'Collaborative': '#3498db',  # Blue
        'Original': '#e74c3c'    # Red
    }
    
    # Helper function to create bar traces
    def create_bar(values, row, col, title, is_percentage=False):
        for agent_type, value in values.items():
            fig.add_trace(
                go.Bar(
                    name=agent_type,
                    x=[agent_type],
                    y=[value],
                    marker_color=colors[agent_type],
                    text=[f'{value:.2f}{"%" if is_percentage else ""}'],
                    textposition='auto',
                    showlegend=(row == 1 and col == 1)  # Only show legend for first subplot
                ),
                row=row,
                col=col
            )
    
    # Create all bar plots
    create_bar({k: v['Time Steps'] for k, v in summary_data.items()}, 1, 1, 'Time Steps')
    create_bar({k: v['Total Moves'] for k, v in summary_data.items()}, 1, 2, 'Total Moves')
    create_bar({k: v['Clean %'] for k, v in summary_data.items()}, 2, 1, 'Clean %', True)
    create_bar({k: v['Success Rate'] * 100 for k, v in summary_data.items()}, 2, 2, 'Success Rate', True)
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text='Comparative Analysis Summary',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        barmode='group'
    )
    
    # Update y-axes with range and labels
    fig.update_yaxes(title_text="Time Steps", range=[0, None], row=1, col=1)
    fig.update_yaxes(title_text="Number of Moves", range=[0, None], row=1, col=2)
    fig.update_yaxes(title_text="Clean Percentage", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 100], row=2, col=2)
    
    # Add helpful annotations
    annotations = [
        dict(
            text="Lower values indicate better performance",
            xref="paper", yref="paper",
            x=0.25, y=1.1,
            showarrow=False,
            font=dict(size=12, color="gray")
        ),
        dict(
            text="Lower values indicate better performance",
            xref="paper", yref="paper",
            x=0.75, y=1.1,
            showarrow=False,
            font=dict(size=12, color="gray")
        ),
        dict(
            text="Higher values indicate better performance",
            xref="paper", yref="paper",
            x=0.25, y=0.45,
            showarrow=False,
            font=dict(size=12, color="gray")
        ),
        dict(
            text="Higher values indicate better performance",
            xref="paper", yref="paper",
            x=0.75, y=0.45,
            showarrow=False,
            font=dict(size=12, color="gray")
        )
    ]
    
    for annot in annotations:
        fig.add_annotation(annot)
    
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
    
    # Get agent type comparison results
    agent_type_results = compare_agent_types(width, height, max_time, num_trials)
    
    # Create and display all visualizations
    plot_all_metrics(base_results, agent_results, dirty_results)
    plot_performance_metrics(agent_type_results)
    plot_efficiency_metrics(agent_type_results)
    plot_agent_comparison(agent_type_results)
    plot_comparative_summary(agent_type_results)  # Add the new summary plot

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
    
    # Get agent type comparison results
    agent_type_results = compare_agent_types(width, height, max_time, num_trials)
    
    # Create and display all visualizations
    plot_all_metrics(base_results, agent_results, dirty_results)
    plot_performance_metrics(agent_type_results)
    plot_efficiency_metrics(agent_type_results)
    plot_agent_comparison(agent_type_results)
    plot_comparative_summary(agent_type_results)  # Add the new summary plot
