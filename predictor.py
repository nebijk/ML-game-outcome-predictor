import pandas as pd


games = pd.read_csv("games.csv",index_col=0)
print(games.head())

print(games["team"].value_counts())
print(games[games["team"] == "Liverpool"])
print(games["round"].value_counts())
print(games.dtypes)

games["date"] = pd.to_datetime(games["date"])
print(games.dtypes)

games["arena_code"] = games["venue"].astype("category").cat.codes
print(games)
games["vs_code"] = games["opponent"].astype("category").cat.codes
print(games)

games["hour"] = games["time"].str.replace(":.+","", regex=True).astype("int")
games["day_code"] = games["date"].dt.day_of_week
print(games)