"""
Run this script repeatedly to sample different Assault frames.
Each run uses a random seed. The seed is printed so you can reproduce a good frame.

Usage:
    python generate_assault_preview.py
    python generate_assault_preview.py --seed 42   # reproduce a specific frame
    python generate_assault_preview.py --steps 200  # control how many steps to simulate
"""

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from ocatari.core import OCAtari

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='Random seed (random if not set)')
parser.add_argument('--steps', type=int, default=250, help='Number of steps to simulate')
args = parser.parse_args()

seed = args.seed if args.seed is not None else random.randint(0, 100000)
print(f"Using seed: {seed}  (re-run with --seed {seed} to reproduce)")

np.random.seed(seed)
env = OCAtari("ALE/Assault-v5", mode="ram", render_mode="rgb_array", hud=True)
env.reset(seed=seed)

for _ in range(args.steps):
    action = np.random.choice([1, 2, 3, 4, 5], p=[0.25, 0.2, 0.2, 0.175, 0.175])
    _, _, done, trunc, _ = env.step(action)
    if done or trunc:
        env.reset()

frame = env.render()
env.close()

fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(frame)
ax.axis('off')
plt.tight_layout(pad=0)
output = 'assault_preview.png'
plt.savefig(output, dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Saved {output}")
