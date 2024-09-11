import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load the CSV file
matches = pd.read_csv("Games.csv", index_col=0)

# Prepare data
matches["date"] = pd.to_datetime(matches["date"])
matches["target"] = (matches["result"] == "W").astype("int")

# Convert categorical variables
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek

# Create training and test data
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

# Define predictors
predictors = ["venue_code", "opp_code", "hour", "day_code"]

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

# Make predictions
preds = rf.predict(test[predictors])

# Calculate accuracy and precision
accuracy = accuracy_score(test["target"], preds)
precision = precision_score(test["target"], preds)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")

# Combine actual and predicted values
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))

# Rolling averages for specific teams
grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City").sort_values("date")

# Function for rolling averages
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Call the rolling averages function
rolling_averages(group, cols, new_cols)

# Apply the rolling averages function to all teams
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')

# Reset the index
matches_rolling.index = range(matches_rolling.shape[0])

# Function to make predictions
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error

combined, error = make_predictions(matches_rolling, predictors + new_cols)

print(f"New precision score after rolling averages: {error}")

# Combine actual and predicted values with match data
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
print(combined.head(10))

# Function to handle missing team names
class MissingDict(dict):
    __missing__ = lambda self, key: key

# Mapping for teams
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)
combined["new_team"] = combined["team"].map(mapping)

# Merge data to compare home and away team predictions
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

print(merged)
# Filter matches where one team's prediction is a win and the other team's is a loss
result = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()

print(result)