import numpy as np


def iou(x: np.ndarray, y: np.ndarray) -> np.float64:
    scores = []
    classes = np.unique(y)

    for c in classes:
        x_inds = (x == c)
        y_inds = (y == c)

        intersection = np.logical_and(x_inds, y_inds).sum()
        union = np.logical_or(x_inds, y_inds).sum()

        if union == 0:
            scores.append(1.0)
        else:
            scores.append(intersection / union)
    return np.mean(scores)


if __name__ == "__main__":
    x = np.array([[1, 1, 0], [0, 2, 2], [0, 0, 3]])
    y = np.array([[1, 1, 0], [0, 2, 2], [0, 0, 1]])
    print(f"IoU: {iou(x, y):.4f}")
    # Output: IoU: 0.6667
