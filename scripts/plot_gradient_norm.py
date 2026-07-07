"""Plot the Riemannian gradient norm over inner iterations (Figure 8 of the paper)."""

import matplotlib.pyplot as plt
import numpy as np

# Iteration x-axis
x = np.arange(1, 6)

# Dictionary of gradient norms for each starting value
data = {
    1000: [39.9375, 0, 0, 0, 0],
    800: [4.7265625, 11.46875, 1.427734375, 0.89794921875, 0.71875],
    600: [0.91943359375, 0.78515625, 0.70458984375, 0.5537109375, 0.487548828125],
    400: [0.319091796875, 0.248046875, 0.1895751953125, 0.1787109375, 0.1690673828125],
    200: [0.252685546875, 0.288818359375, 0.355224609375, 0.1256103515625, 0.10479736328125],
}

# Optional: manually define color palette
colors = ["#90C9E7", '#219EBC', '#136783', '#02304A', "#0B1C25"][::-1]  # or use seaborn / matplotlib colormaps

# Plot
plt.figure(figsize=(12, 5))

for i, (start_value, y) in enumerate(sorted(data.items(), reverse=True)):
    plt.plot(
        x,
        (np.array(y) * 5) ** 0.5,
        linewidth=2,
        label=f"{start_value}",
        color=colors[i],  # specify color here
    )

# Labels
plt.xlabel(r"Guidance inner iteration $K$", fontsize=24, labelpad=10)
plt.ylabel(r"$\sqrt{\|\eta_t^{(k)} \operatorname{grad}_{\mathcal{S}_{r, t}} f\|_2}$", fontsize=24, labelpad=10)

# Ticks
plt.xticks(ticks=x, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(alpha=0.3, axis='y')

# Axis styling
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x')
ax.tick_params(axis='y')

# Legend
plt.legend(title=r"Timestep", fontsize=12, title_fontsize=12, loc='upper right')

# Save and show
plt.tight_layout()
plt.savefig("assets/gradient_norm.pdf", dpi=1200, bbox_inches='tight', transparent=True)
plt.show()

