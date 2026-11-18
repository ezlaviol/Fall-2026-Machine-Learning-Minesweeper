import numpy as np
from preprocessing import TILE_COLUMNS, TILE_ROWS, TOTAL_MINES


UNSAFE_TILES = ['Unchecked', 'Unchecked Question']
FLAGGED_TILES = ['Flagged']

DEBUG_CSP = False

# Analyzes all the neighbors of a given tile and returns safety information
# Returns a tuple (totalSafety, remaining_mines, unflagged_neighbors, neighbors)
# totalSafety: number of unflagged neighbors minus remaining mines
# remaining_mines: number of mines still to be flagged around this tile
# unflagged_neighbors: list of indices of unflagged neighboring tiles
# neighbors: list of [state, index] for all neighboring tiles
def neighbor_analysis(tile, grid, tile_number, rows=TILE_ROWS, cols=TILE_COLUMNS):
    neighbors = []
    row = tile_number // cols
    col = tile_number % cols
    
    unflagged_neighbors = []
    flagged_count = 0

    # check all possible neighbors
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                neighbor_idx = r * cols + c
                neighbor_state = grid[neighbor_idx]
                neighbors.append([neighbor_state, neighbor_idx])
                
                if neighbor_state in UNSAFE_TILES:
                    unflagged_neighbors.append(neighbor_idx)
                elif neighbor_state in FLAGGED_TILES:
                    flagged_count += 1

    remaining_mines = int(tile) - flagged_count
    totalSafety = len(unflagged_neighbors) - remaining_mines

    return totalSafety, remaining_mines, unflagged_neighbors, neighbors


# Runs multi-pass inference on a virtual grid copy to maximize safe moves per frame
def analyze_grid_with_csp(grid_state, total_mines=TOTAL_MINES, rows=TILE_ROWS, cols=TILE_COLUMNS):
    virtual_grid = list(grid_state)
    safeFlags = set()
    safePicks = set()
    invalidFlags = set()

    # 1. SELF-HEALING PASS: Detect and remove impossible flags (remaining_mines < 0)
    for i in range(len(virtual_grid)):
        if virtual_grid[i] in ['1', '2', '3', '4', '5', '6', '7', '8']:
            _, remaining_mines, _, neighbors = neighbor_analysis(
                virtual_grid[i], virtual_grid, i
            )
            if remaining_mines < 0:
                if DEBUG_CSP:
                    print(f"Conflict at Tile {i} ({virtual_grid[i]}): too many adjacent flags!")
                for state, n_idx in neighbors:
                    if state in FLAGGED_TILES:
                        invalidFlags.add(n_idx)
                        virtual_grid[n_idx] = 'Unchecked'

    # 2. TOTAL FLAG CAP: Never exceed total_mines across existing + new flags
    existing_valid_flags = sum(1 for tile in virtual_grid if tile in FLAGGED_TILES)
    max_new_flags = max(0, total_mines - existing_valid_flags)

    def can_add_flag():
        return len(safeFlags) < max_new_flags

    pass_num = 0
    progress_made = True

    while progress_made:
        progress_made = False
        pass_num += 1

        if DEBUG_CSP:
            print(f"\n--- CSP Inference Pass {pass_num} ---")

        for i in range(len(virtual_grid)):
            if virtual_grid[i] in ['1', '2', '3', '4', '5', '6', '7', '8']:
                totalSafety, remaining_mines, unflagged_neighbors, neighbors = neighbor_analysis(
                    virtual_grid[i], virtual_grid, i
                )

                if not unflagged_neighbors:
                    continue

                # Rule 1: Unknown neighbors == remaining mines -> Flag (up to flag cap)
                if totalSafety == 0:
                    for neighbor_idx in unflagged_neighbors:
                        if neighbor_idx not in safeFlags and can_add_flag():
                            safeFlags.add(neighbor_idx)
                            virtual_grid[neighbor_idx] = 'Flagged'
                            progress_made = True
                    if DEBUG_CSP:
                        print(f"Tile {i} ({virtual_grid[i]}) -> Flagged: {unflagged_neighbors}")

                # Rule 2: Remaining mines == 0 -> Safe to pick
                elif remaining_mines == 0:
                    for neighbor_idx in unflagged_neighbors:
                        if neighbor_idx not in safePicks:
                            safePicks.add(neighbor_idx)
                            virtual_grid[neighbor_idx] = 'Checked'
                            progress_made = True
                    if DEBUG_CSP:
                        print(f"Tile {i} ({virtual_grid[i]}) -> Safe Pick: {unflagged_neighbors}")

        # 3. SUBSET CONSTRAINT PASS
        if not progress_made:
            active_clues = []
            for i in range(len(virtual_grid)):
                if virtual_grid[i] in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    _, rem_mines, unflagged, _ = neighbor_analysis(virtual_grid[i], virtual_grid, i)
                    if unflagged and rem_mines > 0:
                        active_clues.append((set(unflagged), rem_mines))

            for idx1 in range(len(active_clues)):
                set1, mines1 = active_clues[idx1]
                for idx2 in range(len(active_clues)):
                    if idx1 == idx2:
                        continue
                    set2, mines2 = active_clues[idx2]

                    if set1 < set2:
                        diff_set = set2 - set1
                        diff_mines = mines2 - mines1

                        if diff_mines == 0:
                            for tile_idx in diff_set:
                                if tile_idx not in safePicks:
                                    safePicks.add(tile_idx)
                                    virtual_grid[tile_idx] = 'Checked'
                                    progress_made = True

                        elif diff_mines == len(diff_set):
                            for tile_idx in diff_set:
                                if tile_idx not in safeFlags and can_add_flag():
                                    safeFlags.add(tile_idx)
                                    virtual_grid[tile_idx] = 'Flagged'
                                    progress_made = True

        # 4. GLOBAL ENDGAME PASS
        if not progress_made:
            flagged_total = sum(1 for tile in virtual_grid if tile in FLAGGED_TILES)
            unchecked_indices = [i for i, tile in enumerate(virtual_grid) if tile in UNSAFE_TILES]
            mines_left = total_mines - flagged_total

            if unchecked_indices:
                if mines_left == 0:
                    for tile_idx in unchecked_indices:
                        if tile_idx not in safePicks:
                            safePicks.add(tile_idx)
                            virtual_grid[tile_idx] = 'Checked'
                            progress_made = True

                elif mines_left == len(unchecked_indices):
                    for tile_idx in unchecked_indices:
                        if tile_idx not in safeFlags and can_add_flag():
                            safeFlags.add(tile_idx)
                            virtual_grid[tile_idx] = 'Flagged'
                            progress_made = True

    return list(safeFlags), list(safePicks), list(invalidFlags)
