import pandas as pd


games = pd.read_csv("games.csv",index_col=0)
print(games.head())

print(games["team"].value_counts())
print(games[games["team"] == "Liverpool"])
print(games["round"].value_counts())
print(games.dtypes)