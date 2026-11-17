import numpy as np
from preprocessing import TILE_COLUMNS, TILE_ROWS


UNSAFE_TILES = ['Unchecked', 'Unchecked Question']
FLAGGED_TILES = ['Flagged']

DEBUG_CSP = True


def neighbor_analysis(tile, grid, tile_number):
    neighbors = []
    row = tile_number // TILE_COLUMNS
    col = tile_number % TILE_COLUMNS
    
    unflagged_neighbors = []
    flagged_count = 0

    # Check all 8 possible neighbors
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < TILE_ROWS and 0 <= c < TILE_COLUMNS:
                neighbor_idx = r * TILE_COLUMNS + c
                neighbor_state = grid[neighbor_idx]
                neighbors.append([neighbor_state, neighbor_idx])
                
                if neighbor_state in UNSAFE_TILES:
                    unflagged_neighbors.append(neighbor_idx)
                elif neighbor_state in FLAGGED_TILES:
                    flagged_count += 1

    # Remaining mines around this tile = tile number - already flagged neighbors
    remaining_mines = int(tile) - flagged_count
    
    # Difference between unknown neighbors and remaining mines
    totalSafety = len(unflagged_neighbors) - remaining_mines

    if DEBUG_CSP:
        print(f"\nTile: {tile}\tTile Number: {tile_number}")
        print(f"row {row} col {col}\nNeighbors: {neighbors}")
        print(f"Unflagged: {len(unflagged_neighbors)}\tRemaining Mines: {remaining_mines}\tTotal Safety: {totalSafety}")
        
    return totalSafety, remaining_mines, unflagged_neighbors


# Returns deduplicated lists of grid indices: (safe_to_flag, safe_to_pick)
def analyze_grid_with_csp(grid_state):
    safeFlags = set()  # grid indices of tiles safe to flag
    safePicks = set()  # grid indices of tiles safe to click

    for i in range(len(grid_state)):
        if grid_state[i] in ['1', '2', '3', '4', '5', '6', '7', '8']:
            totalSafety, remaining_mines, unflagged_neighbors = neighbor_analysis(
                grid_state[i], grid_state, i
            )
            
            # Skip if there are no unknown neighbors left to act on
            if not unflagged_neighbors:
                continue

            # Rule 1: Number of unknown neighbors equals remaining mines -> ALL unknown neighbors are mines
            if totalSafety == 0:
                for neighbor_idx in unflagged_neighbors:
                    safeFlags.add(neighbor_idx)
                if DEBUG_CSP:
                    print(f"Tile {i} -> Flag neighbors: {unflagged_neighbors}")

            # Rule 2: All mines around this tile are already flagged -> ALL unknown neighbors are safe to pick
            elif remaining_mines == 0:
                for neighbor_idx in unflagged_neighbors:
                    safePicks.add(neighbor_idx)
                if DEBUG_CSP:
                    print(f"Tile {i} -> Pick neighbors: {unflagged_neighbors}")

    return list(safeFlags), list(safePicks)


# def neighbor_analysis(tile, grid, tile_number):
#     neighbors = []
#     # Implement neighbor analysis logic here
#     row = tile_number // TILE_COLUMNS
#     col = tile_number % TILE_COLUMNS
#     baseSafety = 0  # incremented for each unsafe neighbor

#     # Check all 8 possible neighbors
#     for dr in [-1, 0, 1]:
#         for dc in [-1, 0, 1]:
#             if dr == 0 and dc == 0:
#                 continue
#             r, c = row + dr, col + dc
#             if 0 <= r < TILE_ROWS and 0 <= c < TILE_COLUMNS:
#                 neighbors.append([grid[(r * TILE_COLUMNS + c)], 
#                                   (r * TILE_COLUMNS + c)])

#     for n in neighbors:
#         if n[0] in UNSAFE_TILES:
#             baseSafety += 1

#     if int(tile) in range(1, 10):
#         totalSafety = baseSafety - int(tile)

#     if DEBUG_CSP:
#         print(f"\nTile: {tile}\tTile Number: {tile_number}")
#         print(f"row {row} col {col}\nNeighbors: {neighbors}")
#         print(f"Base Safety: {baseSafety}\tTotal Safety: {totalSafety}")
#     return totalSafety, neighbors

# # Returns a list of safe moves based on the current grid state
# def analyze_grid_with_csp(grid_state):
#     tileInfo = []   # log all tiles analyzed for neighbor safety
#     # based on neighbor analysis...
#     safeFlags = []  # tiles whose neighbors should be flagged
#     safePicks = []  # tiles whose neighbors can be safely clicked 

#     # Implement CSP logic here
#     for i in range(len(grid_state)):
#         for j in range(1, 10):
#             if grid_state[i] == str(j):
#                 tileInfo.append(neighbor_analysis(
#                     grid_state[i], grid_state, i))
#                 if tileInfo[-1][0] == 0:
#                     safeFlags.append([i, tileInfo[-1][1]])
#                     if DEBUG_CSP:
#                         print(f"Tile {i}'s neighbors should be flagged.")
#                         print(f"Safety: {tileInfo[-1][0]}\nNeighbors: {tileInfo[-1][1]}")
#                 elif tileInfo[-1][0] < 0:
#                     safePicks.append([i, tileInfo[-1][1]])
#                     if DEBUG_CSP:
#                         print(f"Tile {i} has some safe neighbors.")
#                         print(f"Safety: {tileInfo[-1][0]}\nNeighbors: {tileInfo[-1][1]}")

#     # print(f"Safe moves found: {len(safeFlags)}")
#     return safeFlags, safePicks