# ----- IMPORTS -----
import cv2, numpy as np     # OpenCV (image reading) and NumPy (array maths)
from pathlib import Path    # For file paths
from PIL import Image
import pyautogui
import pygetwindow as gw
import ctypes
from ctypes import wintypes


# ----- CONSTANTS -----
DEBUG_VERBOSE = False
DEBUG_DICTIONARY = True
DEBUG_IMAGE = True
DEBUG_CLASSIFY = True

GRID_SIDES = 34
GRID_TOP = 270
GRID_BOTTOM = 34
TILE_WIDTH = 32
TILE_HEIGHT = 32
TILE_COLUMNS = 9
TILE_ROWS = 9

DWMWA_EXTENDED_FRAME_BOUNDS = 9

FOLDER_TILES = Path('./tiles/')    # points to the tiles folder
FOLDER_GRIDS = Path('./grids/')    # points to the grids folder

TARGET_WORD = "uncut"  # find all images with this keyword
TARGET_WORD_PROCESSED = "cut"
TARGET_APPLICATION = "XP Minesweeper Classic"
TARGET_FILE_PREFIX = f"{TARGET_WORD} gamestate "
TARGET_FILE_FORMAT = ".png"


# ----- FUNCTIONS -----
def get_visible_bounds(win):
    rect = wintypes.RECT()
    result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(win._hWnd),
        DWMWA_EXTENDED_FRAME_BOUNDS,
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if result != 0:
        raise OSError(f"DwmGetWindowAttribute failed: {result}")
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top

def capture_gamestate(number):
    windows = gw.getWindowsWithTitle(TARGET_APPLICATION)
    if not windows:
        raise RuntimeError(f"The {TARGET_APPLICATION} window not found")

    win = windows[0]
    print(f"Bring {TARGET_APPLICATION} to the foreground. Capturing in 5 seconds...")
    pyautogui.sleep(5)

    if win.isMinimized:
        raise RuntimeError(f"{TARGET_APPLICATION} is minimized; restore it before capture")

    screenshot = pyautogui.screenshot(
        region=get_visible_bounds(win)
    )
    screenshot.save(f"grids/{TARGET_WORD} gamestate {number}.png")

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


def crop_window(image, image_path):
    image_path = Path(image_path)

    croppedImage = image.crop((
        GRID_SIDES,
        GRID_TOP,
        image.width - GRID_SIDES,
        image.height - GRID_BOTTOM
    ))

    if not image_path.name.startswith(TARGET_WORD):
        raise ValueError(f"Filename must start with {TARGET_WORD!r}: {image_path}")

    newName = image_path.name.replace(TARGET_WORD, TARGET_WORD_PROCESSED, 1)
    outputPath = image_path.with_name(newName)

    croppedImage.save(outputPath)
    if DEBUG_IMAGE and DEBUG_VERBOSE:
        croppedImage.show()

    return croppedImage

# Process a single grid image given its prefix and count, 
# returning the cropped and converted OpenCV image.
def process_grid(count):
    gridName = f"{TARGET_FILE_PREFIX}{count}{TARGET_FILE_FORMAT}"
    gridPath = f"{FOLDER_GRIDS}/{gridName}"
    myScreenshotGrid = crop_window(Image.open(gridPath), gridPath)
    # Convert the Pillow image to a 3-channel OpenCV image.
    myScreenshotGrid = cv2.cvtColor(
        np.array(myScreenshotGrid.convert("RGB")),
        cv2.COLOR_RGB2BGR
    )
    return myScreenshotGrid


def classify_grid(grid, templates):
    screenshot_grid = process_grid(grid)
    tiles = split_grid(np.array(screenshot_grid), columns=TILE_COLUMNS, rows=TILE_ROWS)
    gridState = []
    for row in tiles:
        for tile in row:
            tile_resized = cv2.resize(tile, (16, 16), interpolation=cv2.INTER_AREA)
            gridState.append(classify(tile_resized, templates))  # Classify each tile in the grid
    if DEBUG_CLASSIFY: print(gridState)  # Print the classified grid state after processing all images
    return gridState


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

capture_gamestate(0)

gridStates = []

# Process each grid image
for i in range(len(list(FOLDER_GRIDS.glob(f"{TARGET_FILE_PREFIX}*")))):
    gridStates.append(classify_grid(i, templates))

# tile = cv2.resize(tiles[0][0], (16, 16), interpolation=cv2.INTER_AREA)
# print(classify(tile, templates))  # Classify the top-left tile