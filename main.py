# ----- IMPORTS -----
import cv2, numpy as np     # OpenCV (image reading) and NumPy (array maths)
from pathlib import Path    # For file paths
from PIL import Image


# ----- CONSTANTS -----
DEBUG_VERBOSE = False
DEBUG_DICTIONARY = True
DEBUG_IMAGE = True

GRID_SIDES = 34
GRID_TOP = 270
GRID_BOTTOM = 34
TILE_WIDTH = 32
TILE_HEIGHT = 32
TILE_COLUMNS = 9
TILE_ROWS = 9

FOLDER_TILES = Path('./tiles/')    # points to the tiles folder
FOLDER_GRIDS = Path('./grids/')    # points to the grids folder


# ----- FUNCTIONS -----
def classify(tile, myTemplates):
    return min(myTemplates, key=lambda n: np.sum((myTemplates[n].astype(int) - tile.astype(int))**2))
# Given a cropped tile image, classify does the following:

# 1. Loops over every template name n.
# 2. Computes the sum of squared differences (SSD) between that template and the tile.
# It subtracts pixel by pixel, squares the differences, and adds them up. 
# A score of 0 means identical, and larger scores mean less similar.
# 3. min(..., key=...) returns the name with the lowest score, i.e. the closest match.

def split_grid(image, columns, rows):
    height, width = image.shape[:2]

    if width % columns != 0 or height % rows != 0:
        raise ValueError("Image dimensions must divide evenly into the grid")

    tile_width = width // columns
    tile_height = height // rows

    tiles = []
    for row in range(rows):
        tile_row = []
        for column in range(columns):
            x = column * tile_width
            y = row * tile_height

            tile = image[y:y + tile_height, x:x + tile_width]
            tile_row.append(tile)

        tiles.append(tile_row)

    return tiles

# def crop_window(image, image_path):     # crops the window down to just the grid
#     croppedImage = image.crop((GRID_SIDES, GRID_TOP, image.width - GRID_SIDES, image.height - GRID_BOTTOM))
#     if DEBUG_IMAGE:
#         croppedImage.show()  # for debugging, shows the cropped grid window
#     croppedImage.save(f"{image.name} cropped.png")
#     return croppedImage

def crop_window(image, image_path):
    image_path = Path(image_path)

    croppedImage = image.crop((
        GRID_SIDES,
        GRID_TOP,
        image.width - GRID_SIDES,
        image.height - GRID_BOTTOM
    ))

    output_path = image_path.with_name(
            f"{image_path.stem} cropped{image_path.suffix}"
    )

    if DEBUG_IMAGE:
        croppedImage.save(output_path)
        if DEBUG_VERBOSE:
            croppedImage.show()

    return croppedImage


# ----- MAIN CODE -----
tileNames = []              # empty list of file names, to be populated

for item in FOLDER_TILES.iterdir():
    if item.is_file() and "Minesweeper Tile" in item.name:
        if DEBUG_DICTIONARY and DEBUG_VERBOSE:
            print(item.name)
            print((((item.name).replace("Minesweeper Tile", "")).replace(".png", "")).strip())
        tileNames.append((((item.name).replace("Minesweeper Tile", "")).replace(".png", "")).strip())
#   looks for files starting with "Minesweeper Tile", then turns them into class names

if DEBUG_DICTIONARY:
    print(tileNames)

# list of class names 
templates = {name: cv2.imread(f"tiles/Minesweeper Tile {name}.png") for name in tileNames}
# Builds a dictionary that maps each class name (e.g. "1", "2", "flag", "unrevealed")
# to its reference image. cv2.imread loads each file as a NumPy array of pixels.

# Check that all templates were loaded successfully
for name, image in templates.items():
    if image is None:
        raise FileNotFoundError(f"Could not load template for {name}")

if DEBUG_DICTIONARY and DEBUG_VERBOSE:
    print(templates)

gridName = 'uncut test grid.png'
gridPath = f"{FOLDER_GRIDS}/{gridName}"
screenshot_grid = crop_window(Image.open(gridPath), gridPath)
# Convert the Pillow image to a 3-channel OpenCV image.
screenshot_grid = cv2.cvtColor(
    np.array(screenshot_grid.convert("RGB")),
    cv2.COLOR_RGB2BGR
)

tiles = split_grid(np.array(screenshot_grid), columns=TILE_COLUMNS, rows=TILE_ROWS)
tile = cv2.resize(tiles[0][0], (16, 16), interpolation=cv2.INTER_AREA)
print(classify(tile, templates))  # Classify the top-left tile