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
DEBUG_DICTIONARY = False
DEBUG_IMAGE = False
DEBUG_CLASSIFY = False

ctypes.windll.user32.SetProcessDPIAware()
DPI = ctypes.windll.user32.GetDpiForSystem() // 96  # Evaluates to 1 at 100%, 2 at 200%
GRID_SCALE = 2  # change this to 2 if set to x2 or 1 if not

TITLE_BAR_HEIGHT = 31 * DPI
GRID_SIDES = (17 * GRID_SCALE - (GRID_SCALE - 1)) * DPI
GRID_TOP = TITLE_BAR_HEIGHT + (104 * GRID_SCALE * DPI)
GRID_BOTTOM = GRID_SIDES
TILE_WIDTH = 16 * DPI * GRID_SCALE
TILE_HEIGHT = 16 * DPI * GRID_SCALE

TILE_COLUMNS = 9
TILE_ROWS = 9
TOTAL_MINES = 10

DWMWA_EXTENDED_FRAME_BOUNDS = 9

FOLDER_TILES = Path('./tiles/')    # points to the tiles folder
FOLDER_GRIDS = Path('./grids/')    # points to the grids folder
FOLDER_GRIDS.mkdir(parents=True, exist_ok=True)

TARGET_WORD = "uncut"  # find all images with this keyword
TARGET_WORD_PROCESSED = "cut"
TARGET_APPLICATION = "XP Minesweeper Classic"
TARGET_FILE_PREFIX = f"{TARGET_WORD} gamestate "
TARGET_FILE_FORMAT = ".png"


# ----- FUNCTIONS -----
# Retrieves the visible bounds of a window using the DWM API.
# Returns a tuple (left, top, width, height) of the visible window bounds. 
# Used by capture_gamestate to determine the region to capture.
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

# Finds the Minesweeper window, restores it if minimized, and brings it to the foreground.
def get_game_window():
    # Filter out small dialog boxes (like the New Record Time popup) so we always grab the main board
    windows = [
        w for w in gw.getWindowsWithTitle(TARGET_APPLICATION)
        if w.width > 150 and w.height > 200
    ]
    if not windows:
        raise RuntimeError(f"The {TARGET_APPLICATION} window not found")

    win = windows[0]
    hwnd = wintypes.HWND(win._hWnd)
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # 1. Restore if minimized
    if win.isMinimized:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE = 9
        pyautogui.sleep(0.15)

    # 2. Bypass Windows focus-stealing prevention
    foreground_hwnd = user32.GetForegroundWindow()
    if foreground_hwnd != hwnd:
        foreground_thread = user32.GetWindowThreadProcessId(foreground_hwnd, None)
        current_thread = kernel32.GetCurrentThreadId()

        # Attach input threads so Windows allows focus transfer
        if foreground_thread != current_thread:
            user32.AttachThreadInput(foreground_thread, current_thread, True)

        # Sending an Alt key event unlocks SetForegroundWindow on Windows 10/11
        pyautogui.press('alt')
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)

        if foreground_thread != current_thread:
            user32.AttachThreadInput(foreground_thread, current_thread, False)

        pyautogui.sleep(0.25)

    return win

# Captures the game interface automatically, saving it as a screenshot with the given number.
def capture_gamestate(number):
    FOLDER_GRIDS.mkdir(parents=True, exist_ok=True)
    win = get_game_window()
    win_left, win_top, _, _ = get_visible_bounds(win)
    pyautogui.moveTo(win_left + 10, win_top + 10)  # Park cursor off the grid

    screenshot = pyautogui.screenshot(
        region=get_visible_bounds(win)
    )
    screenshot.save(f"grids/{TARGET_WORD} gamestate {number}.png")


# Given a cropped tile image, classify does the following:

# 1. Loops over every template name n.
# 2. Computes the sum of squared differences (SSD) between that template and the tile.
# It subtracts pixel by pixel, squares the differences, and adds them up. 
# A score of 0 means identical, and larger scores mean less similar.
# 3. min(..., key=...) returns the name with the lowest score, i.e. the closest match.
def classify(tile, myTemplates):
    return min(myTemplates, key=lambda n: np.sum((myTemplates[n].astype(int) - tile.astype(int))**2))

# Splits an image into a grid of tiles with the specified number of columns and rows.
# Returns a 2D list of tiles, where each sublist represents a row of tiles.
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

# Crops the window image to the game grid area, based on predefined grid boundaries.
# Returns the cropped image as a Pillow Image object.
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

# Takes an unclassified, raw grid index and a dictionary of templates,
# returns the classified grid state as a list of tile classes.
def classify_grid(grid, templates, rows=TILE_ROWS, columns=TILE_COLUMNS):
    screenshot_grid = process_grid(grid)
    tiles = split_grid(np.array(screenshot_grid), columns=columns, rows=rows)
    gridState = []
    for row in tiles:
        for tile in row:
            tile_resized = cv2.resize(tile, (16, 16), interpolation=cv2.INTER_AREA)
            gridState.append(classify(tile_resized, templates))  # Classify each tile in the grid
    if DEBUG_CLASSIFY: print(gridState)  # Print the classified grid state after processing all images
    return gridState

def from_gamestate_to_grid(number, templates, capture_first=True, rows=TILE_ROWS, columns=TILE_COLUMNS):
    if capture_first:
        capture_gamestate(number)
    return classify_grid(number, templates, rows, columns)