import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
IMAGE_PATH = "cameraman.png"
PI = np.pi
SPLIT_RATIO = 10
N_DEGREES = 20
N_SAMPLES = N_DEGREES**2 * SPLIT_RATIO
DEGREES = np.arange(1, int(N_DEGREES * 1.42))
OUTPUT_DIR = "fig"

# Load image
image = plt.imread(IMAGE_PATH)


def f2(x, y):
    """Image function to aproximate."""
    min_dim = 256
    y_idx = np.int32((1 - y / PI) * (min_dim - 1))
    x_idx = np.int32(x / PI * (min_dim - 1))
    return image[y_idx, x_idx]


def f1(x, y):
    """Synthetic function to approximate."""
    if (x - 2.0) ** 2 + (y - 1.9) ** 2 < 0.8:
        return 1
    elif (x - 1.9) ** 2 + (y - 2) ** 2 < 1:
        return 0
    elif abs(x - 0.6) < 0.2 and abs(y - 0.9) < 0.6:
        return 1
    elif (3.14 - x) + (y) < 0.4:
        return 1
    return 0


def generate_data(n_samples):
    """Generate random samples points."""
    X, y = [], []
    for _ in range(n_samples):
        x_coord = np.random.rand() * PI
        y_coord = np.random.rand() * PI
        X.append([x_coord, y_coord])
        y.append(f1(x_coord, y_coord))
    return X, y


def cosine_transform(x, degree):
    """2D Cosine transform for basis expansion."""
    i_values = np.arange(degree)
    j_values = np.arange(degree)
    cos_i = np.cos(np.outer(i_values, [x_i[0] for x_i in x]))
    cos_j = np.cos(np.outer(j_values, [x_i[1] for x_i in x]))
    X_cosine = []
    for k in range(len(x)):
        cos_i_k = cos_i[:, k]
        cos_j_k = cos_j[:, k]
        x_cosine = np.outer(cos_i_k, cos_j_k).ravel()
        X_cosine.append(x_cosine)
    return np.array(X_cosine)


def plot_train_points(f, X_train, y_train):
    """Plot training points."""
    plt.figure(figsize=(8, 8))
    X_train = np.array(X_train)
    plt.scatter(
        X_train[:, 0], X_train[:, 1], color=[(i, i, i) for i in y_train], marker="o"
    )
    plt.xlim(0, PI)
    plt.ylim(0, PI)
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/train_points.png", facecolor="grey")
    plt.close()


def plot_function(f):
    """Plot the original function."""
    x = np.linspace(0, PI, 256)
    y = np.linspace(0, PI, 256)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(f)(X, Y)
    plt.figure(figsize=(8, 8))
    plt.imshow(
        Z, extent=(0, PI, 0, PI), origin="lower", cmap="gray", interpolation="bicubic"
    )
    plt.clim(0, 1)
    plt.axis("off")
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    # save with transparent background
    plt.savefig(f"{OUTPUT_DIR}/function.png", transparent=True)
    plt.close()


def plot_degree_image(model, degree, suffix, normalize=True):
    """Plot and save the approximation of the function for a given degree."""
    x = np.linspace(0, PI, 256)
    y = np.linspace(0, PI, 256)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    Z = model.predict(cosine_transform(xy, degree)).reshape(X.shape)
    plt.figure(figsize=(8, 8))
    plt.imshow(
        Z, extent=(0, PI, 0, PI), origin="lower", cmap="gray", interpolation="bilinear"
    )
    plt.axis("off")
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    if normalize:
        plt.clim(0, 1)
    else:
        plt.clim(-500, 500)
    plt.savefig(f"{OUTPUT_DIR}/{degree}{suffix}.png", transparent=True)
    plt.close()


def plot_errors(
    degrees, train_errors, test_errors, train_errors_reg, test_errors_reg, train_samples
):
    """Plot training and test errors."""
    plt.figure(figsize=(10, 6), dpi=300)
    plt.semilogy(
        degrees**2 / train_samples, test_errors, label="Error de Testeo", color="black"
    )
    plt.semilogy(
        degrees**2 / train_samples,
        test_errors_reg,
        label="Error de Testeo Regularizado",
        color="orange",
    )
    plt.semilogy(
        degrees**2 / train_samples,
        train_errors,
        label="Error de Entrenamiento",
        color="black",
        linestyle="--",
    )
    plt.axvline(x=1, color="gray", linestyle="--", label="Umbral de interpolación")

    plt.xlabel("Capacidad del modelo")
    plt.ylabel("Error cuadrático medio")
    plt.ylim(2e-4, max(test_errors))
    plt.gca().spines[["top", "right"]].set_visible(False)

    plt.gca().spines["left"].set_position(("data", 0))
    plt.legend()
    plt.savefig("double_descent_2D.png")
    plt.show()


def main():
    X, y = generate_data(N_SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - 1 / SPLIT_RATIO, random_state=42
    )
    train_samples = len(X_train)

    plot_function(f1)
    plot_train_points(f1, X_train, y_train)

    train_errors, test_errors = [], []
    train_errors_reg, test_errors_reg = [], []
    for degree in tqdm(DEGREES):
        X_cosine = cosine_transform(X_train, degree)
        model = LinearRegression().fit(X_cosine, y_train)
        model_reg = RidgeCV(alphas=np.logspace(-6, 6, 6)).fit(X_cosine, y_train)

        if degree % 1 == 0:
            plot_degree_image(model, degree, "")
            plot_degree_image(model_reg, degree, "_reg")

            plot_degree_image(model, degree, "_comp", normalize=False)
            plot_degree_image(model_reg, degree, "_reg_comp", normalize=False)

        train_errors.append(np.mean((model.predict(X_cosine) - y_train) ** 2))
        train_errors_reg.append(np.mean((model_reg.predict(X_cosine) - y_train) ** 2))

        X_cosine = cosine_transform(X_test, degree)
        test_errors.append(np.mean((model.predict(X_cosine) - y_test) ** 2))
        test_errors_reg.append(np.mean((model_reg.predict(X_cosine) - y_test) ** 2))

    plot_errors(
        DEGREES,
        train_errors,
        test_errors,
        train_errors_reg,
        test_errors_reg,
        train_samples,
    )


if __name__ == "__main__":
    main()
