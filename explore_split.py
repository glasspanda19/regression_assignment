"""Load used_car_sales.csv, summarize it, and save a fixed 70/30 split."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SOURCE_PATH = DATA_DIR / "used_car_sales.csv"
TRAIN_PATH = DATA_DIR / "train_cars.csv"
TEST_PATH = DATA_DIR / "test_cars.csv"

# Show every column and print floats in decimal form.
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", "{:.4f}".format)

df = pd.read_csv(SOURCE_PATH)

print("df.shape")
print(df.shape)
print()
print("df.dtypes")
print(df.dtypes)
print()
print("df.describe()")
print(df.describe())
print()

train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)

overlap = train_df.index.intersection(test_df.index)
print(f"index overlap count: {len(overlap)}")
if len(overlap) != 0:
    raise SystemExit(f"train_df and test_df share indexes: {overlap.tolist()}")
print("zero index overlap: True")
print()


def print_sale_price_summary(name, split_df):
    prices = split_df["Sale_Price"]
    print(f"{name} rows: {len(split_df)}")
    print(f"{name} Sale_Price mean: {prices.mean():.4f}")
    print(f"{name} Sale_Price min: {prices.min():.4f}")
    print(f"{name} Sale_Price max: {prices.max():.4f}")


print_sale_price_summary("train_df", train_df)
print()
print_sale_price_summary("test_df", test_df)
print()

if len(train_df) != 168 or len(test_df) != 72:
    raise SystemExit(
        f"expected 168 train rows and 72 test rows, got {len(train_df)} and {len(test_df)}"
    )

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
print(f"saved {TRAIN_PATH.name} ({len(train_df)} rows)")
print(f"saved {TEST_PATH.name} ({len(test_df)} rows)")
