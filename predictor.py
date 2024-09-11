import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

games["goal"]= (games["result"] == "W").astype("int")
randF=RandomForestClassifier(n_estimators=50,min_samples_split=10, random_state=1)
train=games[games["date"]<'2022-01-01']
test=games[games["date"]>'2022-01-01']

predictors=["arena_code", "vs_code","hour","day_code"]
randF.fit(train[predictors],train["goal"])

preds = randF.predict(test[predictors])
acc=accuracy_score(test["goal"], preds)

print(acc)