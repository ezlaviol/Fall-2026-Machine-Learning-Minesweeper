import numpy as np
import pyautogui
from preprocessing import (
    GRID_SIDES,
    GRID_TOP,
    GRID_BOTTOM,
    TILE_COLUMNS,
    TILE_ROWS,
    get_visible_bounds,
    split_grid,
    get_game_window,
)

CLICK_DELAY = 0.02
pyautogui.PAUSE = CLICK_DELAY

SMILEY_OFFSET_ABOVE_GRID = 60  # Pixels above GRID_TOP where the smiley button sits

# Maps every tile index (0..80) to its exact (x, y) screen center.
def build_tile_centers(win):
    win_left, win_top, win_width, win_height = get_visible_bounds(win)

    grid_left = win_left + GRID_SIDES
    grid_top = win_top + GRID_TOP
    grid_width = win_width - (2 * GRID_SIDES)
    grid_height = win_height - GRID_TOP - GRID_BOTTOM

    xs = np.arange(grid_left, grid_left + grid_width)
    ys = np.arange(grid_top, grid_top + grid_height)
    coord_grid = np.stack(np.meshgrid(xs, ys), axis=-1)

    tile_slices = split_grid(coord_grid, columns=TILE_COLUMNS, rows=TILE_ROWS)

    centers = []
    for row in tile_slices:
        for tile_coords in row:
            center_x = int(tile_coords[:, :, 0].mean())
            center_y = int(tile_coords[:, :, 1].mean())
            centers.append((center_x, center_y))

    return centers


# Right-clicks each tile index in the list to place or remove a flag.
def flag_tiles(tile_indices, centers=None):
    if not tile_indices:
        return

    if centers is None:
        win = get_game_window()
        centers = build_tile_centers(win)

    for idx in tile_indices:
        x, y = centers[idx]
        pyautogui.rightClick(x, y)

# Left-clicks each tile index in the list to reveal safe tiles.
def pick_tiles(tile_indices, centers=None):
    if not tile_indices:
        return

    if centers is None:
        win = get_game_window()
        centers = build_tile_centers(win)

    for idx in tile_indices:
        x, y = centers[idx]
        pyautogui.leftClick(x, y)

# Computes window bounds once and executes unflags, flags, and safe picks.
def execute_moves(safe_flags, safe_picks, invalid_flags=None):
    if not safe_flags and not safe_picks and not invalid_flags:
        return

    win = get_game_window()
    centers = build_tile_centers(win)

    if invalid_flags:
        flag_tiles(invalid_flags, centers=centers)
    if safe_flags:
        flag_tiles(safe_flags, centers=centers)
    if safe_picks:
        pick_tiles(safe_picks, centers=centers)


# Clicks the centered smiley face button above the grid to reset the board.
def click_smiley():
    win = get_game_window()
    win_left, win_top, win_width, _ = get_visible_bounds(win)

    smiley_x = win_left + (win_width // 2)
    smiley_y = win_top + GRID_TOP - SMILEY_OFFSET_ABOVE_GRID

    pyautogui.click(smiley_x, smiley_y)
    pyautogui.sleep(0.25)
