import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

grid_size = 7
num_targets = 5

stick_colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple', 'deeppink', 'teal', 'saddlebrown', 'mediumvioletred', 'steelblue', 'olivedrab', 'coral']
directions = [(-1,0), (1,0), (0,-1), (0,1)]

cat_expressions = ['😸', '😺', '😼', '😽', '🙀', '😹', '😻', '😾', '😿', '🐱', '😺', '😸']

def draw_stick_figure(ax, cx, cy, color, lw=2.0, expression_idx=0):
    expr = cat_expressions[expression_idx % len(cat_expressions)]
    ax.text(cx, cy, expr, fontsize=42, ha='center', va='center', zorder=5,
            color=color, fontfamily='DejaVu Sans')

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
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')

# Inner gridlines in grey
for i in range(grid_size + 1):
    ax.axhline(i - 0.5, color='#AAAAAA', linewidth=1.0)
    ax.axvline(i - 0.5, color='#AAAAAA', linewidth=1.0)

# Black border
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)

for i, (row, col) in enumerate(target_positions):
    color = stick_colors[i % len(stick_colors)]
    draw_stick_figure(ax, col, row, color, expression_idx=i)
    dr, dc = random.choice(directions)
    mid_col = col + dc * 0.65
    mid_row = row + dr * 0.65
    ax.annotate('', xy=(mid_col + dc * 0.30, mid_row + dr * 0.30),
                xytext=(mid_col - dc * 0.30, mid_row - dr * 0.30),
                arrowprops=dict(arrowstyle='->', color=color, lw=4.0,
                                mutation_scale=20))

def draw_robot(ax, cx, cy, s=0.55):
    import matplotlib.patches as mpatches
    # Antenna
    ax.plot([cx, cx], [cy - s*1.05, cy - s*0.72], color='#444', linewidth=2, zorder=4)
    ax.add_patch(plt.Circle((cx, cy - s*1.12), s*0.08, facecolor='#FF4444', edgecolor='#444', linewidth=1, zorder=5))
    # Head
    ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.42, cy - s*0.70), s*0.84, s*0.60,
                 boxstyle='round,pad=0.02', facecolor='#A8C8E8', edgecolor='#444', linewidth=2, zorder=4))
    # Eyes
    for ex in [cx - s*0.15, cx + s*0.15]:
        ax.add_patch(plt.Circle((ex, cy - s*0.42), s*0.10, facecolor='#1144AA', edgecolor='#444', linewidth=1, zorder=5))
        ax.add_patch(plt.Circle((ex + s*0.03, cy - s*0.44), s*0.04, facecolor='white', linewidth=0, zorder=6))
    # Mouth
    ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.18, cy - s*0.22), s*0.36, s*0.10,
                 boxstyle='round,pad=0.01', facecolor='#1144AA', edgecolor='#444', linewidth=1, zorder=5))
    # Body
    ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.46, cy + s*0.02), s*0.92, s*0.68,
                 boxstyle='round,pad=0.02', facecolor='#88AACC', edgecolor='#444', linewidth=2, zorder=4))
    # Chest panel
    ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.26, cy + s*0.12), s*0.52, s*0.36,
                 boxstyle='round,pad=0.01', facecolor='#CCDDE8', edgecolor='#666', linewidth=1, zorder=5))
    # Chest buttons
    for bx, bc in [(cx - s*0.10, '#FF4444'), (cx + s*0.10, '#44CC44')]:
        ax.add_patch(plt.Circle((bx, cy + s*0.30), s*0.07, facecolor=bc, edgecolor='#444', linewidth=1, zorder=6))
    # Arms
    for side in [-1, 1]:
        ax.plot([cx + side*s*0.46, cx + side*s*0.72], [cy + s*0.18, cy + s*0.38],
                color='#88AACC', linewidth=6, solid_capstyle='round', zorder=3)
        ax.plot([cx + side*s*0.46, cx + side*s*0.72], [cy + s*0.18, cy + s*0.38],
                color='#444', linewidth=2, solid_capstyle='round', zorder=3)


def draw_coffee_cup(ax, cx, cy):
    draw_robot(ax, cx, cy)

draw_coffee_cup(ax, agent_pos[1], agent_pos[0])

# Tick styling
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.tick_params(colors='black', labelsize=9)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('black')

ax.invert_yaxis()

plt.tight_layout()
plt.savefig('grid_preview.png', dpi=120, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("Saved grid_preview.png")
