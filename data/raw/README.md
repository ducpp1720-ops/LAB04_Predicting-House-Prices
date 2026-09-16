# Dataset

Place the two real Google MLCC California Housing CSV files in this directory:

- `california_housing_train.csv` — 17,000 rows
- `california_housing_test.csv` — 3,000 rows

Both files must contain:
`longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value`.

No synthetic fallback is used. The training script fails explicitly when either file is absent.
