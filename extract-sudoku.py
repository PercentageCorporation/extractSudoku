import cv2
import numpy as np
import shutil
import tkinter as tk
from tkinter import filedialog as fd
import os
import sys

#IMAGE = "SudokuType3.png"
#IMAGE = "ss2740.png"
#IMAGE = "ss2740-50.png"
#IMAGE = "ss2740-150.png"
#IMAGE = "solver.png"
#IMAGE = "htcol.png"
#IMAGE = "xwing.png"

root = tk.Tk()
root.withdraw()
start = os.path.abspath("assets")

imgfilename = fd.askopenfilename(
    title="Select Image File",
    initialdir='../games',
    filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
    )
#print("filename:",filename)
if len(imgfilename) == 0:
    sys.exit()

IMAGE = os.path.basename(imgfilename)
print("Selected file:", IMAGE)

OUTPUT = "cells"

def load_training():

    samples = []
    labels = []

    for value in range(1, 10):

        path = f"training/{value}"

        for filename in os.listdir(path):

            img = cv2.imread(
                os.path.join(path, filename),
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            samples.append(
                img.flatten()
            )

            labels.append(value)

    return (
        np.array(samples, dtype=np.float32),
        np.array(labels, dtype=np.int32)
    )

def normalize_digit(inner, component, size=40):
    x, y, w, h, area = component

    digit = inner[y:y+h, x:x+w]

    target_h = 34

    scale = target_h / h

    new_w = round(w * scale)
    new_h = target_h

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    result = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    xoff = (size - new_w) // 2
    yoff = (size - new_h) // 2

    result[
        yoff:yoff + new_h,
        xoff:xoff + new_w
    ] = digit

    return result

def digit_difference(a, b):
    return np.mean(cv2.absdiff(a, b))

def digit_similarity(a, b):
    result = cv2.matchTemplate(
        a,
        b,
        cv2.TM_CCOEFF_NORMED
    )

    return result[0][0]

def recognize_digit(digit, samples, labels):

    sample = digit.flatten().astype(np.float32)

    differences = samples - sample

    distances = np.sum(
        differences * differences,
        axis=1
    )

    digit_distances = []

    for value in range(1, 10):

        mask = labels == value

        best_distance = np.min(
            distances[mask]
        )

        digit_distances.append(
            (best_distance, value)
        )

    digit_distances.sort()

    best_distance, best_value = \
        digit_distances[0]

    second_distance, second_value = \
        digit_distances[1]

    return (
        best_value,
        best_distance,
        second_value,
        second_distance
    )

#################################################################################################

if os.path.exists("digits"):
    shutil.rmtree("digits")

os.makedirs("digits")

training_samples, training_labels = load_training()

print(f"Loaded {len(training_labels)} training digits")

path = f"../games/{IMAGE}"
print("Reading: ", path)
img = cv2.imread(path)

if img is None:
    raise RuntimeError(f"Could not read {IMAGE}")

print("Image size:", img.shape)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray.png", gray)

os.makedirs(OUTPUT, exist_ok=True)

# Convert grayscale to binary image
binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

cv2.imwrite("binary.png", binary)
height, width = gray.shape

cell_w = width / 9
cell_h = height / 9

os.makedirs("digits", exist_ok=True)
game = []

for row in range(9):
    game_row = []

    for col in range(9):

        x1 = round(col * cell_w)
        x2 = round((col + 1) * cell_w)

        y1 = round(row * cell_h)
        y2 = round((row + 1) * cell_h)

        cell_height = y2 - y1
        cell_width = x2 - x1

        margin_y = round(cell_height * 0.05)
        margin_x = round(cell_width * 0.05)

        inner = binary[
            y1 + margin_y:y2 - margin_y,
            x1 + margin_x:x2 - margin_x
        ]

        num_labels, component_labels, stats, centroids = \
            cv2.connectedComponentsWithStats(
                inner,
                connectivity=8
            )

        large_component = None

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            ih, iw = inner.shape

            # Ignore anything touching an edge
            if (
                x <= 2 or
                y <= 2 or
                x + w >= iw - 2 or
                y + h >= ih - 2
            ):
                continue

            if (row, col) in [
                    (2, 0),   # r3c1
                    (3, 3),   # r4c4
                    (5, 2),   # r6c3
                    (6, 6),   # r7c7
                    (7, 4)    # r8c5                
                ]:
                print(
                    f"r{row+1}c{col+1} "
                    f"w={w} h={h} area={area} "
                    f"hRatio={h/ih:.3f} "
                    f"areaRatio={area/(iw*ih):.3f}"
                )                

            # Large game digit rather than candidate
            if (
                h >= ih * 0.30 and
                w >= iw * 0.20
            ):
                large_component = (
                    x, y, w, h, area
                )
                break

        if large_component:
            digit = normalize_digit(
                inner,
                large_component
            )

            value, distance, second, second_distance = \
                recognize_digit(
                    digit,
                    training_samples,
                    training_labels
                )

            print(
                f"r{row+1}c{col+1}: "
                f"{value} "
                f"d={distance:.0f} "
                f"second={second} "
                f"d={second_distance:.0f} "
                f"ratio={distance/second_distance:.3f}"
            )
            cv2.imwrite(
                f"digits/r{row+1}c{col+1}.png",
                digit
            )

            game_row.append(value)

        else:
            game_row.append(0)

    game.append(game_row)

print()

for row in game:
    print(row)

linear = ""
for row in range(9):
    for col in range(9):
        linear += str(game[row][col]);

print(linear)

gfile = os.path.splitext(IMAGE)[0] + ".game"
path = f"../games/{gfile}"
print("Saving game:", path)
with open(path, 'w', encoding='utf-8') as f:
    f.write(linear)

