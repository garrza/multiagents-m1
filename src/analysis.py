from typing import Dict, List
from src.simulation import Simulation


def run_experiment(
    width: int,
    height: int,
    num_agents: int,
    dirty_percentage: float,
    max_time: int,
    num_trials: int,
    agent_type: str = "mixed"
) -> List[Dict]:
    results = []
    for _ in range(num_trials):
        sim = Simulation(width, height, num_agents, dirty_percentage, max_time, agent_type)
        stats = sim.run()
        stats["agent_type"] = agent_type  # Add agent type to statistics
        results.append(stats)
    return results


# Example usage and analysis
def analyze_results(results: List[Dict], max_time: int) -> None:
    # Group results by agent type if multiple types exist
    agent_types = set(r["agent_type"] for r in results)
    
    for agent_type in agent_types:
        type_results = [r for r in results if r["agent_type"] == agent_type]
        print(f"\nResults for {agent_type} agents:")
        
        # Calculate success probabilities for different time thresholds
        time_thresholds = [0.25, 0.50, 0.75, 1.0]
        for threshold in time_thresholds:
            threshold_time = max_time * threshold
            success_count = sum(
                1
                for r in type_results
                if r["time_taken"] <= threshold_time and r["clean_percentage"] == 100
            )
            probability = success_count / len(type_results)
            print(
                f"Probability of success within {threshold*100}% of max time: {probability:.2f}"
            )

        # Calculate averages
        avg_time = sum(r["time_taken"] for r in type_results) / len(type_results)
        avg_moves = sum(r["total_moves"] for r in type_results) / len(type_results)
        avg_clean = sum(r["clean_percentage"] for r in type_results) / len(type_results)

        print(f"\nAverage Statistics:")
        print(f"Time taken: {avg_time:.2f} steps")
        print(f"Moves made: {avg_moves:.2f}")
        print(f"Clean percentage: {avg_clean:.2f}%")


def compare_agent_types(
    width: int,
    height: int,
    num_agents: int,
    dirty_percentage: float,
    max_time: int,
    num_trials: int
) -> None:
    """Compare different agent type configurations"""
    # Run experiments for different agent types
    agent_types = ["mixed", "collaborative", "original"]
    all_results = {}
    
    for agent_type in agent_types:
        results = run_experiment(
            width, height, num_agents, dirty_percentage, max_time, num_trials, agent_type
        )
        all_results[agent_type] = results
    
    # Print comparative analysis
    print("\nComparative Analysis:")
    print("-" * 50)
    
    for agent_type, results in all_results.items():
        avg_time = sum(r["time_taken"] for r in results) / len(results)
        avg_moves = sum(r["total_moves"] for r in results) / len(results)
        avg_clean = sum(r["clean_percentage"] for r in results) / len(results)
        success_rate = sum(1 for r in results if r["clean_percentage"] == 100) / len(results)
        
        print(f"\n{agent_type.capitalize()} Configuration:")
        print(f"Average time: {avg_time:.2f} steps")
        print(f"Average moves: {avg_moves:.2f}")
        print(f"Average clean percentage: {avg_clean:.2f}%")
        print(f"Success rate: {success_rate:.2f}")
