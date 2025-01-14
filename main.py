import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.simulation import Simulation
from src.analysis import run_experiment, analyze_results, compare_agent_types
from src.visualization import create_performance_dashboard, plot_agent_comparison


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


def run_comparative_analysis(width, height, max_time, num_trials):
    """Run a comparative analysis of different simulation configurations.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        max_time: Maximum simulation time
        num_trials: Number of trials to run
    """
    # Test different numbers of agents
    agent_counts = [2, 4, 6, 8]
    dirty_percentage = 0.5
    
    # Results for varying number of agents
    agent_results = {}
    print("\nTesting different numbers of agents...")
    for num_agents in agent_counts:
        print(f"Running trials with {num_agents} agents...")
        results = run_experiment(
            width, height, num_agents, dirty_percentage, max_time, num_trials
        )
        agent_results[num_agents] = {
            'times': [r['time_taken'] for r in results],
            'moves': [r['total_moves'] for r in results],
            'clean': [r['clean_percentage'] for r in results]
        }
    
    # Test different dirty percentages
    dirty_percentages = [0.3, 0.5, 0.7, 0.9]
    num_agents = 6
    
    # Results for varying dirty percentages
    dirty_results = {}
    print("\nTesting different initial dirty percentages...")
    for dirty_pct in dirty_percentages:
        print(f"Running trials with {dirty_pct*100}% initial dirty cells...")
        results = run_experiment(
            width, height, num_agents, dirty_pct, max_time, num_trials
        )
        dirty_results[dirty_pct] = {
            'times': [r['time_taken'] for r in results],
            'moves': [r['total_moves'] for r in results],
            'clean': [r['clean_percentage'] for r in results]
        }
    
    return agent_results, dirty_results


def main():
    """Main entry point for running the simulation analysis."""
    width = 20
    height = 20
    max_time = 1000
    num_trials = 30  # Reduced trials since we're running multiple configurations
    
    # Run base configuration analysis
    print("Running base configuration analysis...")
    base_results = run_experiment(
        width, height, 6, 0.5, max_time, num_trials
    )
    
    # Run comparative analysis
    agent_results, dirty_results = run_comparative_analysis(
        width, height, max_time, num_trials
    )
    
    print("\nGenerating performance analysis plots...")
    dashboard = create_performance_dashboard(base_results, agent_results, dirty_results)
    dashboard.show()
    
    # Compare different agent types
    print("\nComparing different agent types...")
    compare_agent_types(width, height, 6, 0.5, max_time, num_trials)


if __name__ == "__main__":
    main()
