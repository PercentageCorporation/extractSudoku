import cv2
import numpy as np
import os

def recognize_digit(digit, samples, labels, k=3):

    sample = digit.flatten().astype(np.float32)

    differences = samples - sample

    distances = np.sum(
        differences * differences,
        axis=1
    )

    nearest = np.argsort(distances)[:k]

    nearest_labels = labels[nearest]
    nearest_distances = distances[nearest]

    values, counts = np.unique(
        nearest_labels,
        return_counts=True
    )

    value = values[
        np.argmax(counts)
    ]

    return (
        int(value),
        nearest_labels,
        nearest_distances
    )

###################################################################################################

samples = []
labels = []

for value in range(1, 10):
    path = f"training/{value}"

    for filename in os.listdir(path):
        img = cv2.imread(
            os.path.join(path, filename),
            cv2.IMREAD_GRAYSCALE
        )

        samples.append(
            img.flatten()
        )

        labels.append(value)

samples = np.array(
    samples,
    dtype=np.float32
)

labels = np.array(
    labels,
    dtype=np.float32
)

print("samples:", samples.shape)
print("labels:", labels.shape)

for value in range(1, 10):
    print(
        value,
        np.count_nonzero(labels == value)
    )

