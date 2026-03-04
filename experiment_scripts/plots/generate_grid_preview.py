import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

grid_size = 7
num_targets = 5

stick_colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple']
directions = [(-1,0), (1,0), (0,-1), (0,1)]

def draw_stick_figure(ax, cx, cy, color, lw=2.0):
    ax.add_patch(plt.Circle((cx, cy - 0.22), 0.12, facecolor=color, edgecolor=color, linewidth=1))
    ax.plot([cx, cx], [cy - 0.10, cy + 0.12], color=color, linewidth=lw)
    ax.plot([cx - 0.22, cx + 0.22], [cy - 0.00, cy - 0.00], color=color, linewidth=lw)
    ax.plot([cx, cx - 0.18], [cy + 0.12, cy + 0.35], color=color, linewidth=lw)
    ax.plot([cx, cx + 0.18], [cy + 0.12, cy + 0.35], color=color, linewidth=lw)

def sample_spread_positions(grid_size, n, min_dist=2, seed=42):
    rng = random.Random(seed)
    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    chosen = []
    rng.shuffle(all_cells)
    for cell in all_cells:
        if all(abs(cell[0] - p[0]) >= min_dist or abs(cell[1] - p[1]) >= min_dist for p in chosen):
            chosen.append(cell)
        if len(chosen) == n:
            break
    return chosen

positions = sample_spread_positions(grid_size, num_targets + 1, min_dist=2)
agent_pos = positions[0]
target_positions = positions[1:]

fig, ax = plt.subplots(figsize=(8, 8))

# Parchment background
fig.patch.set_facecolor('#F2E0C0')
ax.set_facecolor('#F2E0C0')

ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')

# Rustic gridlines in warm brown
for i in range(grid_size + 1):
    ax.axhline(i - 0.5, color='#8B5E3C', linewidth=1.5, alpha=0.6)
    ax.axvline(i - 0.5, color='#8B5E3C', linewidth=1.5, alpha=0.6)

# Thick rustic border
for spine in ax.spines.values():
    spine.set_edgecolor('#5C3317')
    spine.set_linewidth(4)

for i, (row, col) in enumerate(target_positions):
    color = stick_colors[i % len(stick_colors)]
    draw_stick_figure(ax, col, row, color)
    dr, dc = random.choice(directions)
    mid_col = col + dc * 0.65
    mid_row = row + dr * 0.65
    ax.annotate('', xy=(mid_col + dc * 0.30, mid_row + dr * 0.30),
                xytext=(mid_col - dc * 0.30, mid_row - dr * 0.30),
                arrowprops=dict(arrowstyle='->', color=color, lw=4.0,
                                mutation_scale=20))

def draw_coffee_cup(ax, cx, cy, cup_color='#C0522A', rim_color='#7A2E0E', steam_color='#A0522A', s=0.78):
    import matplotlib.patches as mpatches
    # Steam lines
    for sx, phase in [(-0.08*s, 0), (0.08*s, np.pi)]:
        t = np.linspace(0, 1, 30)
        wx = cx + sx + 0.04*s * np.sin(phase + t * 3 * np.pi)
        wy = cy - 0.32*s - t * 0.22*s
        ax.plot(wx, wy, color=steam_color, linewidth=2, alpha=0.7)
    # Saucer
    ax.add_patch(mpatches.Ellipse((cx, cy + 0.30*s), width=0.72*s, height=0.14*s,
                                   facecolor=rim_color, edgecolor=rim_color, zorder=2))
    ax.add_patch(mpatches.Ellipse((cx, cy + 0.28*s), width=0.58*s, height=0.09*s,
                                   facecolor=cup_color, edgecolor=cup_color, zorder=3))
    # Cup body (trapezoid)
    ax.add_patch(plt.Polygon([
        (cx - 0.27*s, cy - 0.22*s), (cx + 0.27*s, cy - 0.22*s),
        (cx + 0.20*s, cy + 0.22*s), (cx - 0.20*s, cy + 0.22*s),
    ], closed=True, facecolor=cup_color, edgecolor=rim_color, linewidth=2, zorder=4))
    # Rim (top ellipse)
    ax.add_patch(mpatches.Ellipse((cx, cy - 0.22*s), width=0.54*s, height=0.10*s,
                                   facecolor=rim_color, edgecolor=rim_color, zorder=5))
    # Inner top (lighter)
    ax.add_patch(mpatches.Ellipse((cx, cy - 0.22*s), width=0.46*s, height=0.07*s,
                                   facecolor='#6B2A00', edgecolor='#6B2A00', zorder=6))
    # Handle
    ax.add_patch(mpatches.Arc((cx + 0.27*s, cy + 0.02*s), width=0.22*s, height=0.28*s,
                               angle=0, theta1=-90, theta2=90,
                               color=rim_color, linewidth=4, zorder=4))

draw_coffee_cup(ax, agent_pos[1], agent_pos[0])

# Tick styling
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.tick_params(colors='#5C3317', labelsize=9)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('#5C3317')
    label.set_fontfamily('serif')

ax.invert_yaxis()

plt.tight_layout()
plt.savefig('grid_preview.png', dpi=120, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("Saved grid_preview.png")
