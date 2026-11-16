from preprocessing import FOLDER_TILES, FOLDER_GRIDS, TARGET_FILE_PREFIX, DEBUG_DICTIONARY, DEBUG_VERBOSE
from preprocessing import from_gamestate_to_grid, cv2

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
    gridStates.append(from_gamestate_to_grid(i, templates))
# Each grid image is processed and classified, 
# and the resulting grid states are stored in gridStates.

# tile = cv2.resize(tiles[0][0], (16, 16), interpolation=cv2.INTER_AREA)
# print(classify(tile, templates))  # Classify the top-left tile