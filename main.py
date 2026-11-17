from preprocessing import FOLDER_TILES, FOLDER_GRIDS, TARGET_FILE_PREFIX
from preprocessing import DEBUG_DICTIONARY, DEBUG_VERBOSE
from preprocessing import TILE_COLUMNS, TILE_ROWS
from preprocessing import from_gamestate_to_grid, cv2
from csp import analyze_grid_with_csp, DEBUG_CSP

# ----- MAIN CODE -----
tileNames = []              # empty list of file names, to be populated

# Populate the list of tile class names by scanning the tiles folder 
# for files matching the pattern "Minesweeper Tile *.png".
# The resulting list of names will be used to build the template dictionary.
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

gridStates = []

# Process each grid image
for i in range(len(list(FOLDER_GRIDS.glob(f"{TARGET_FILE_PREFIX}*")))):
    gridStates.append(from_gamestate_to_grid(i, templates, True))
    # Each grid image is processed and classified, 
    # and the resulting grid states are stored in gridStates.
    if DEBUG_CSP:
        print(f"Grid {i}:\n{gridStates[i]}")

    # analyze the new grid state with CSP
    safeFlags, safePicks = analyze_grid_with_csp(gridStates[i])
    print(f"Safe Flags: {safeFlags}\nSafe Picks: {safePicks}")
    if DEBUG_CSP:
        debugSafeFlags = []
        for tile_idx in safeFlags:
            row = (tile_idx // TILE_COLUMNS) + 1
            column = (tile_idx % TILE_COLUMNS) + 1
            debugSafeFlags.append([row, column])
        print(f"You may safely flag (row, col): {debugSafeFlags}")

        debugSafePicks = []
        for tile_idx in safePicks:
            row = (tile_idx // TILE_COLUMNS) + 1
            column = (tile_idx % TILE_COLUMNS) + 1
            debugSafePicks.append([row, column])
        print(f"You may safely click (row, col): {debugSafePicks}")


    # if safeFlags:
    #     # flag_tile(safeFlags[0])
    # if not safeFlags:
    #     print(f"No safe moves found for Grid {i}")

    # future psuedocode
    # - if no safe moves found by CSP, apply DQN instead
    # - update the game state based on the chosen move
    # - make a new grid image