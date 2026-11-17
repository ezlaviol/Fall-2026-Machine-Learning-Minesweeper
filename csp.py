import numpy as np
from preprocessing import TILE_COLUMNS, TILE_ROWS

UNSAFE_TILES = ['Unchecked']

DEBUG_CSP = True


def neighbor_analysis(tile, grid, tile_number):
    neighbors = []
    # Implement neighbor analysis logic here
    row = tile_number // TILE_COLUMNS
    col = tile_number % TILE_COLUMNS
    baseSafety = 0  # incremented for each unsafe neighbor

    # Check all 8 possible neighbors
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < TILE_ROWS and 0 <= c < TILE_COLUMNS:
                neighbors.append(grid[(r * TILE_COLUMNS + c)])

    for n in neighbors:
        if n in UNSAFE_TILES:
            baseSafety += 1

    if int(tile) in range(1, 10):
        totalSafety = baseSafety - int(tile)

    if DEBUG_CSP:
        print(f"Tile: {tile}\tTile Number: {tile_number}")
        print(f"row {row} col {col}\nNeighbors: {neighbors}")
        print(f"Base Safety: {baseSafety}\tTotal Safety: {totalSafety}")
    return totalSafety

# Returns a list of safe moves based on the current grid state
def analyze_grid_with_csp(grid_state):
    tileInfo = []   # for later use with dqn mayhaps?
    safeMoves = []  # list of safe tiles to click based on CSP analysis

    # Implement CSP logic here
    for i in range(len(grid_state)):
        for j in range(1, 10):
            if grid_state[i] == str(j):
                tileInfo.append(neighbor_analysis(
                    grid_state[i], grid_state, i))
                if neighbor_analysis(
                    grid_state[i], grid_state, i) == 0:
                    safeMoves.append(i)

    print(f"Safe moves found: {len(safeMoves)}")
    return safeMoves