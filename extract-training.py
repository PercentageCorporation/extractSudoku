import cv2
import os
import numpy as np



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


def extract_training(image_name, game, source_name):

    print("reading", image_name)
    path = f"training/{image_name}"

    img = cv2.imread(path)

    if img is None:
        raise RuntimeError(
            f"Could not read {image_name}"
        )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    height, width = gray.shape

    cell_w = width / 9
    cell_h = height / 9

    for row in range(9):
        for col in range(9):

            # Known value from the text representation
            value = int(game[row][col])

            # Empty cell -- don't need it for training
            if value == 0:
                continue

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

            num_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(
                    inner,
                    connectivity=8
                )

            large_component = None

            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]

                ih, iw = inner.shape

                # Ignore grid lines or anything touching an edge
                if (
                    x <= 2 or
                    y <= 2 or
                    x + w >= iw - 2 or
                    y + h >= ih - 2
                ):
                    continue

                # Large game digit rather than candidate
                if h >= ih * 0.30:
                    large_component = (
                        x, y, w, h, area
                    )
                    break

            if large_component is None:
                print(
                    f"WARNING: no digit found at "
                    f"r{row+1}c{col+1}, "
                    f"expected {value}"
                )

                print("  inner size:", iw, ih)

                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]

                    print(
                        f"  component:"
                        f" x={x}"
                        f" y={y}"
                        f" w={w}"
                        f" h={h}"
                        f" area={area}"
                        f" height ratio={h/ih:.2f}"
                    )

                continue

            digit = normalize_digit(
                inner,
                large_component
            )

            filename = (
                f"training/{value}/"
                f"{source_name}-"
                f"r{row+1}c{col+1}.png"
            )

            cv2.imwrite(filename, digit)

            print(
                f"  {value}: "
                f"r{row+1}c{col+1}"
            )

##########################################################################################
game1 = [
    "714000935",
    "060900804",
    "005000062",
    "500100398",
    "001589206",
    "000003051",
    "000000609",
    "009007023",
    "140090587"
]

game2 = [
    "300700859",
    "800005301",
    "000030200",
    "236897145",
    "549612738",
    "178050692",
    "000070000",
    "900500007",
    "780006003"
]

extract_training("SudokuType3.png", game1, "type3")

extract_training("solver.png", game2, "solver")