from src.simulation import Simulation
from src.analysis import run_experiment, analyze_results
import matplotlib.pyplot as plt
import numpy as np


def plot_all_metrics(base_results, agent_results, dirty_results):
    # Create figure with subplots in a 3x3 grid
    fig = plt.figure(figsize=(20, 15))
    
    # Extract base metrics
    times = [r['time_taken'] for r in base_results]
    moves = [r['total_moves'] for r in base_results]
    clean_percentages = [r['clean_percentage'] for r in base_results]

    # Distribution plots (top row)
    # Time distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(times, bins=20, edgecolor='black')
    ax1.set_title('Distribution of Time Taken', fontsize=12)
    ax1.set_xlabel('Time Steps', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.axvline(np.mean(times), color='r', linestyle='dashed', label='Mean')
    ax1.legend()

    # Moves distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(moves, bins=20, edgecolor='black')
    ax2.set_title('Distribution of Moves Made', fontsize=12)
    ax2.set_xlabel('Number of Moves', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.axvline(np.mean(moves), color='r', linestyle='dashed', label='Mean')
    ax2.legend()

    # Clean percentage distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(clean_percentages, bins=20, edgecolor='black')
    ax3.set_title('Distribution of Clean Percentage', fontsize=12)
    ax3.set_xlabel('Clean Percentage (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.axvline(np.mean(clean_percentages), color='r', linestyle='dashed', label='Mean')
    ax3.legend()

    # Agent comparison plots (middle row)
    agents = sorted(agent_results.keys())
    
    # Time vs Agents
    ax4 = plt.subplot(3, 3, 4)
    avg_times = [np.mean(agent_results[a]['times']) for a in agents]
    std_times = [np.std(agent_results[a]['times']) for a in agents]
    ax4.errorbar(agents, avg_times, yerr=std_times, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax4.set_title('Average Time vs Number of Agents', fontsize=12)
    ax4.set_xlabel('Number of Agents', fontsize=10)
    ax4.set_ylabel('Average Time Steps', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)

    # Moves vs Agents
    ax5 = plt.subplot(3, 3, 5)
    avg_moves = [np.mean(agent_results[a]['moves']) for a in agents]
    std_moves = [np.std(agent_results[a]['moves']) for a in agents]
    ax5.errorbar(agents, avg_moves, yerr=std_moves, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax5.set_title('Average Moves vs Number of Agents', fontsize=12)
    ax5.set_xlabel('Number of Agents', fontsize=10)
    ax5.set_ylabel('Average Moves', fontsize=10)
    ax5.grid(True, linestyle='--', alpha=0.7)

    # Clean percentage vs Agents
    ax6 = plt.subplot(3, 3, 6)
    avg_clean = [np.mean(agent_results[a]['clean']) for a in agents]
    std_clean = [np.std(agent_results[a]['clean']) for a in agents]
    ax6.errorbar(agents, avg_clean, yerr=std_clean, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax6.set_title('Average Clean % vs Number of Agents', fontsize=12)
    ax6.set_xlabel('Number of Agents', fontsize=10)
    ax6.set_ylabel('Average Clean Percentage', fontsize=10)
    ax6.grid(True, linestyle='--', alpha=0.7)

    # Dirty percentage comparison plots (bottom row)
    pcts = sorted(dirty_results.keys())
    
    # Time vs Dirty Percentage
    ax7 = plt.subplot(3, 3, 7)
    avg_times = [np.mean(dirty_results[p]['times']) for p in pcts]
    std_times = [np.std(dirty_results[p]['times']) for p in pcts]
    ax7.errorbar(pcts, avg_times, yerr=std_times, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax7.set_title('Average Time vs Initial Dirty %', fontsize=12)
    ax7.set_xlabel('Initial Dirty Percentage', fontsize=10)
    ax7.set_ylabel('Average Time Steps', fontsize=10)
    ax7.grid(True, linestyle='--', alpha=0.7)

    # Moves vs Dirty Percentage
    ax8 = plt.subplot(3, 3, 8)
    avg_moves = [np.mean(dirty_results[p]['moves']) for p in pcts]
    std_moves = [np.std(dirty_results[p]['moves']) for p in pcts]
    ax8.errorbar(pcts, avg_moves, yerr=std_moves, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax8.set_title('Average Moves vs Initial Dirty %', fontsize=12)
    ax8.set_xlabel('Initial Dirty Percentage', fontsize=10)
    ax8.set_ylabel('Average Moves', fontsize=10)
    ax8.grid(True, linestyle='--', alpha=0.7)

    # Clean percentage vs Dirty Percentage
    ax9 = plt.subplot(3, 3, 9)
    avg_clean = [np.mean(dirty_results[p]['clean']) for p in pcts]
    std_clean = [np.std(dirty_results[p]['clean']) for p in pcts]
    ax9.errorbar(pcts, avg_clean, yerr=std_clean, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax9.set_title('Average Clean % vs Initial Dirty %', fontsize=12)
    ax9.set_xlabel('Initial Dirty Percentage', fontsize=10)
    ax9.set_ylabel('Average Clean Percentage', fontsize=10)
    ax9.grid(True, linestyle='--', alpha=0.7)

    # Add a main title
    plt.suptitle('Vacuum Cleaner Agent Performance Analysis', fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


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
