from src.simulation import Simulation
from src.analysis import run_experiment, analyze_results
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_all_metrics(base_results, agent_results, dirty_results):
    # Create figure with subplots in a 3x3 grid
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

    # Distribution plots (top row)
    # Time distribution
    fig.add_trace(
        go.Histogram(x=times, name='Time Distribution', nbinsx=20),
        row=1, col=1
    )
    fig.add_vline(x=np.mean(times), line_dash="dash", line_color="red", 
                  annotation_text="Mean", row=1, col=1)

    # Moves distribution
    fig.add_trace(
        go.Histogram(x=moves, name='Moves Distribution', nbinsx=20),
        row=1, col=2
    )
    fig.add_vline(x=np.mean(moves), line_dash="dash", line_color="red",
                  annotation_text="Mean", row=1, col=2)

    # Clean percentage distribution
    fig.add_trace(
        go.Histogram(x=clean_percentages, name='Clean % Distribution', nbinsx=20),
        row=1, col=3
    )
    fig.add_vline(x=np.mean(clean_percentages), line_dash="dash", line_color="red",
                  annotation_text="Mean", row=1, col=3)

    # Agent comparison plots (middle row)
    agents = sorted(agent_results.keys())
    
    # Time vs Agents
    avg_times = [np.mean(agent_results[a]['times']) for a in agents]
    std_times = [np.std(agent_results[a]['times']) for a in agents]
    fig.add_trace(
        go.Scatter(x=agents, y=avg_times, mode='markers', name='Average Time', error_y=dict(type='data', array=std_times, visible=True)),
        row=2, col=1
    )

    # Moves vs Agents
    avg_moves = [np.mean(agent_results[a]['moves']) for a in agents]
    std_moves = [np.std(agent_results[a]['moves']) for a in agents]
    fig.add_trace(
        go.Scatter(x=agents, y=avg_moves, mode='markers', name='Average Moves', error_y=dict(type='data', array=std_moves, visible=True)),
        row=2, col=2
    )

    # Clean percentage vs Agents
    avg_clean = [np.mean(agent_results[a]['clean']) for a in agents]
    std_clean = [np.std(agent_results[a]['clean']) for a in agents]
    fig.add_trace(
        go.Scatter(x=agents, y=avg_clean, mode='markers', name='Average Clean %', error_y=dict(type='data', array=std_clean, visible=True)),
        row=2, col=3
    )

    # Dirty percentage comparison plots (bottom row)
    pcts = sorted(dirty_results.keys())
    
    # Time vs Dirty Percentage
    avg_times = [np.mean(dirty_results[p]['times']) for p in pcts]
    std_times = [np.std(dirty_results[p]['times']) for p in pcts]
    fig.add_trace(
        go.Scatter(x=pcts, y=avg_times, mode='markers', name='Average Time', error_y=dict(type='data', array=std_times, visible=True)),
        row=3, col=1
    )

    # Moves vs Dirty Percentage
    avg_moves = [np.mean(dirty_results[p]['moves']) for p in pcts]
    std_moves = [np.std(dirty_results[p]['moves']) for p in pcts]
    fig.add_trace(
        go.Scatter(x=pcts, y=avg_moves, mode='markers', name='Average Moves', error_y=dict(type='data', array=std_moves, visible=True)),
        row=3, col=2
    )

    # Clean percentage vs Dirty Percentage
    avg_clean = [np.mean(dirty_results[p]['clean']) for p in pcts]
    std_clean = [np.std(dirty_results[p]['clean']) for p in pcts]
    fig.add_trace(
        go.Scatter(x=pcts, y=avg_clean, mode='markers', name='Average Clean %', error_y=dict(type='data', array=std_clean, visible=True)),
        row=3, col=3
    )

    # Update layout
    fig.update_layout(
        height=1000,
        width=1500,
        showlegend=False,
        title_text="Vacuum Cleaner Agent Performance Analysis"
    )
    
    # Show the plot
    fig.show()


def run_comparative_analysis(width, height, max_time, num_trials):
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
    plot_all_metrics(base_results, agent_results, dirty_results)


if __name__ == "__main__":
    main()
