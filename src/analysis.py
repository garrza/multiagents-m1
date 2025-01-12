from typing import Dict, List
from src.simulation import Simulation


def run_experiment(
    width: int,
    height: int,
    num_agents: int,
    dirty_percentage: float,
    max_time: int,
    num_trials: int,
) -> List[Dict]:
    results = []
    for _ in range(num_trials):
        sim = Simulation(width, height, num_agents, dirty_percentage, max_time)
        stats = sim.run()
        results.append(stats)
    return results


# Example usage and analysis
def analyze_results(results: List[Dict], max_time: int) -> None:
    # Calculate success probabilities for different time thresholds
    time_thresholds = [0.25, 0.50, 0.75, 1.0]
    for threshold in time_thresholds:
        threshold_time = max_time * threshold
        success_count = sum(
            1
            for r in results
            if r["time_taken"] <= threshold_time and r["clean_percentage"] == 100
        )
        probability = success_count / len(results)
        print(
            f"Probability of success within {threshold*100}% of max time: {probability:.2f}"
        )

    # Calculate averages
    avg_time = sum(r["time_taken"] for r in results) / len(results)
    avg_moves = sum(r["total_moves"] for r in results) / len(results)
    avg_clean = sum(r["clean_percentage"] for r in results) / len(results)

    print(f"\nAverage Statistics:")
    print(f"Time taken: {avg_time:.2f} steps")
    print(f"Moves made: {avg_moves:.2f}")
    print(f"Clean percentage: {avg_clean:.2f}%")
