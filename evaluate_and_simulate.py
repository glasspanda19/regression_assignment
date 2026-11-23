"""Score the v1 OLS model on the test set and simulate auction buying rules."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train_cars.csv"
TEST_PATH = DATA_DIR / "test_cars.csv"
AUCTION_PATH = DATA_DIR / "auction_lots.csv"
PLOT_PATH = DATA_DIR / "test_pred_vs_actual.png"

FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident + Num_Owners"
RECON_COST = 1500


def should_buy(predicted_price, auction_price, recon_cost=1500):
    """Buy when the auction price plus recon cost is below the predicted sale price."""
    return auction_price + recon_cost < predicted_price


def out_of_sample_r_squared(y, y_hat):
    """R-squared against the test-set mean: 1 - SS_res / SS_tot."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    y_test_mean = y.mean()
    residual_sum_squares = np.sum((y - y_hat) ** 2)
    total_sum_squares = np.sum((y - y_test_mean) ** 2)
    return 1 - residual_sum_squares / total_sum_squares


def rmse(y, y_hat):
    """Root mean squared error in the same units as y."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return np.sqrt(np.mean((y - y_hat) ** 2))


def dollar_tick(value, _position):
    return f"${value:,.0f}"


def simulate_rule(auction_df, predicted_prices):
    """Apply should_buy to every lot and summarize cars bought and profit."""
    buy_flags = [
        bool(should_buy(predicted_price, auction_price, RECON_COST))
        for predicted_price, auction_price in zip(
            predicted_prices, auction_df["Auction_Price"]
        )
    ]
    bought = auction_df.loc[buy_flags]
    profit = bought["Later_Sale_Price"] - bought["Auction_Price"] - RECON_COST
    return {
        "cars_bought": int(len(bought)),
        "money_losing_cars": int((profit < 0).sum()),
        "total_profit": float(profit.sum()),
    }


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
auction_df = pd.read_csv(AUCTION_PATH)

v1_model = smf.ols(FORMULA, data=train_df).fit()
baseline_price = float(train_df["Sale_Price"].mean())

y_test = test_df["Sale_Price"].to_numpy(dtype=float)
baseline_test_pred = np.full(len(test_df), baseline_price)
v1_test_pred = v1_model.predict(test_df).to_numpy(dtype=float)

print("Test set evaluation")
print(f"baseline prediction (train Sale_Price mean): ${baseline_price:,.2f}")
print(f"baseline out-of-sample R-squared: {out_of_sample_r_squared(y_test, baseline_test_pred):.4f}")
print(f"baseline RMSE: ${rmse(y_test, baseline_test_pred):,.2f}")
print(f"v1 out-of-sample R-squared: {out_of_sample_r_squared(y_test, v1_test_pred):.4f}")
print(f"v1 RMSE: ${rmse(y_test, v1_test_pred):,.2f}")
print()

axis_min = min(y_test.min(), v1_test_pred.min())
axis_max = max(y_test.max(), v1_test_pred.max())
padding = (axis_max - axis_min) * 0.05
axis_min -= padding
axis_max += padding

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, v1_test_pred, color="#1f77b4", alpha=0.85, label="Test cars", zorder=3)
ax.plot(
    [axis_min, axis_max],
    [axis_min, axis_max],
    linestyle="--",
    color="black",
    linewidth=1.2,
    label="y = x",
    zorder=2,
)
ax.set_xlim(axis_min, axis_max)
ax.set_ylim(axis_min, axis_max)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Test set: Actual vs v1 Predicted Sale Price")
ax.set_xlabel("Actual Sale_Price")
ax.set_ylabel("v1 Predicted Sale_Price")
ax.xaxis.set_major_formatter(FuncFormatter(dollar_tick))
ax.yaxis.set_major_formatter(FuncFormatter(dollar_tick))
ax.legend(frameon=True)
ax.grid(True, linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=150)
plt.close(fig)
print(f"saved {PLOT_PATH.name}")
print()

baseline_auction_pred = np.full(len(auction_df), baseline_price)
v1_auction_pred = v1_model.predict(auction_df).to_numpy(dtype=float)
baseline_result = simulate_rule(auction_df, baseline_auction_pred)
v1_result = simulate_rule(auction_df, v1_auction_pred)
lot_count = len(auction_df)

comparison = pd.DataFrame(
    [
        {
            "rule": "baseline",
            "cars_bought": f"{baseline_result['cars_bought']} / {lot_count}",
            "money_losing_cars": baseline_result["money_losing_cars"],
            "total_profit": f"${baseline_result['total_profit']:,.2f}",
        },
        {
            "rule": "v1",
            "cars_bought": f"{v1_result['cars_bought']} / {lot_count}",
            "money_losing_cars": v1_result["money_losing_cars"],
            "total_profit": f"${v1_result['total_profit']:,.2f}",
        },
    ]
)

print("Auction decision comparison")
print(f"Buy when Auction_Price + ${RECON_COST:,.0f} < predicted Sale_Price")
print("Profit on a bought car = Later_Sale_Price - Auction_Price - "
      f"${RECON_COST:,.0f}")
print(comparison.to_string(index=False))
