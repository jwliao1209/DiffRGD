"""Plot the chi distribution used in the polar-decomposition figure (Figure 2)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi

# Chi distribution parameters
df = 3  # degrees of freedom
x = np.linspace(0, 10, 500)
y = chi.pdf(x, df)

# Update the plot with black color and transparency, and save the figure
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, color='black', linewidth=4, alpha=0.6)  # black with transparency

# Remove all spines and labels
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
plt.tight_layout()

# Save the figure
output_path = "assets/chi_distribution.png"
plt.savefig(output_path, dpi=600, transparent=True)
plt.show()

