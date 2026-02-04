"""Quick evaluation script for trained Assault agents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from assault.assault_experiment import AssaultExperiment


def main():
    AssaultExperiment.evaluate_checkpoint(
        exp_dir="logs/assault_ppo_bidding_exp1_20260203_113532",
        model_filename="assault_agent.pt",
        num_eval_episodes=5,
        num_video_episodes=3,
    )


if __name__ == "__main__":
    main()
