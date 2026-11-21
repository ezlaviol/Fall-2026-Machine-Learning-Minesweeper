import pandas as pd

RESULTS_PATH = './results.csv'

df = pd.read_csv(RESULTS_PATH)
winrate = (df['Outcome'] == "Victory").mean()

print(f"Winrate: {winrate:.2%}") # round two decimal points