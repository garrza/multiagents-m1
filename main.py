import numpy as np
from src.simulation import Simulation
from src.analysis import run_experiment, analyze_results, compare_agent_types
from src.visualization import (
    create_performance_dashboard,
    plot_agent_comparison,
    create_distribution_plots,
    create_agent_comparison_plots,
    create_dirty_comparison_plots,
    plot_all_metrics
)


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
