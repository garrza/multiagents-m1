from src.simulation import Simulation
from src.analysis import run_experiment, analyze_results


def main():
    width = 20
    height = 20
    num_agents = 4
    dirty_percentage = 0.5
    max_time = 1000
    num_trials = 100

    # Run experiments
    results = run_experiment(
        width, height, num_agents, dirty_percentage, max_time, num_trials
    )
    analyze_results(results, max_time)


if __name__ == "__main__":
    main()
