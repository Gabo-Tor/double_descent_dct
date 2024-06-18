import os

import matplotlib.pyplot as plt

FOLDER = "./fig"

# Read numberred images from the folder and generate X
X, X_reg = [], []
test, test_reg = [], []

for file in os.listdir(FOLDER):
    if file.endswith("_comp.png") and file.rstrip("_comp.png").isnumeric():
        X.append(int(file.rstrip("_comp.png"))**2)
        test.append(os.path.getsize(os.path.join(FOLDER, file)))
    if file.endswith("_reg_comp.png"):
        X_reg.append(int(file.rstrip("_reg_comp.png"))**2)
        test_reg.append(os.path.getsize(os.path.join(FOLDER, file)))

# Normalize
X = list(map(lambda x: x / (60**2), X))
X_reg = list(map(lambda x: x / (60**2), X_reg))

min_test = min(test)
test = list(map(lambda x: x / min_test, test))
test_reg = list(map(lambda x: x / min_test, test_reg))

# Create the plot
fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
ax.scatter(X, test, color="black", label="no regularizado", alpha=0.5)
ax.scatter(X_reg, test_reg, color="orange", label="regularizado", alpha=0.5)

# Remove the top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# use log but not use scientific notation in y
ax.set_yscale("log")
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

# Add labels and legend
ax.set_xlabel("Capacidad del modelo")
ax.set_ylabel("Complejidad de la función")
ax.legend()
plt.tight_layout()

plt.savefig("complexity.png", transparent=True)
plt.show()
