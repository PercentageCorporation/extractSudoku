import cv2
import os
import numpy as np
import shutil

IMAGE = "SudokuType3.png"
#IMAGE = "ss2740.png"
#IMAGE = "solverapp.png"
OUTPUT = "cells"

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

def recognize_digit(digit):
    scores = []

    for value, template in digit_templates.items():
        score = digit_similarity(
            digit,
            template
        )

        scores.append([score, value])

    scores.sort(reverse=True)

    best_score, best_value = scores[0]
    second_score, second_value = scores[1]

    return (
        best_value,
        best_score,
        second_value,
        second_score
    )

#################################################################################################

if os.path.exists("digits"):
    shutil.rmtree("digits")

os.makedirs("digits")

img = cv2.imread(IMAGE)

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

template_cells = {
    1: "r1c2",
    2: "r3c9",
    3: "r1c8",
    4: "r1c3",
    5: "r1c9",
    6: "r2c2",
    7: "r1c1",
    8: "r2c7",
    9: "r1c7",
}


os.makedirs("digits", exist_ok=True)
game = []

for row in range(9):
    game_row = []

    for col in range(9):

        x1 = round(col * cell_w)
        x2 = round((col + 1) * cell_w)

        y1 = round(row * cell_h)
        y2 = round((row + 1) * cell_h)

        margin = 5

        inner = binary[
            y1 + margin:y2 - margin,
            x1 + margin:x2 - margin
        ]

        num_labels, labels, stats, centroids = \
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

            if h >= 25:
                large_component = (x, y, w, h, area)
                break

        if large_component:
            digit = normalize_digit(
                inner,
                large_component
            )

            cv2.imwrite(
                f"digits/r{row+1}c{col+1}.png",
                digit
            )

            game_row.append("?")

        else:
            game_row.append(0)

    game.append(game_row)

os.makedirs("templates", exist_ok=True)

for value, name in template_cells.items():
    img = cv2.imread(
        f"digits/{name}.png",
        cv2.IMREAD_GRAYSCALE
    )

    cv2.imwrite(
        f"templates/{value}.png",
        img
    )

print()

digit_templates = {}

for value in range(1, 10):
    digit_templates[value] = cv2.imread(
        f"templates/{value}.png",
        cv2.IMREAD_GRAYSCALE
    )

game = []

for row in range(9):
    game_row = []

    for col in range(9):
        filename = f"digits/r{row+1}c{col+1}.png"

        if not os.path.exists(filename):
            game_row.append(0)
            continue

        digit = cv2.imread(
            filename,
            cv2.IMREAD_GRAYSCALE
        )

        value, score, second, second_score = \
            recognize_digit(digit)

        print(
            f"r{row+1}c{col+1}:",
            f"{value} ({score:.4f})",
            f"next {second} ({second_score:.4f})"
        )

        game_row.append(value);

    game.append(game_row)


print()

for row in game:
    print(row)
