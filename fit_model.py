"""Fit a full OLS model and a significance-trimmed OLS model on train_cars.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train_cars.csv"

PREDICTORS = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
ALPHA = 0.05

train_df = pd.read_csv(TRAIN_PATH)

full_formula = "Sale_Price ~ " + " + ".join(PREDICTORS)
full_model = smf.ols(full_formula, data=train_df).fit()

print("Full model summary")
print(full_model.summary())
print()

print("Full model coefficients, p-values, and fit")
for term, coefficient in full_model.params.items():
    print(f"coefficient {term}: {coefficient:.4f}")
for term, pvalue in full_model.pvalues.items():
    print(f"p-value {term}: {pvalue:.4f}")
print(f"R-squared: {full_model.rsquared:.4f}")
print(f"adjusted R-squared: {full_model.rsquared_adj:.4f}")
print(f"training residual standard error: ${np.sqrt(full_model.mse_resid):,.2f}")
print(f"training RMSE: ${np.sqrt(np.mean(full_model.resid ** 2)):,.2f}")
print()


def predictor_for_term(term):
    """Map a statsmodels term name back to one of the five predictors."""
    if term == "Intercept":
        return None
    for name in PREDICTORS:
        if term == name or term.startswith(f"{name}["):
            return name
    raise ValueError(f"Unrecognized model term: {term}")


dropped = []
for name in PREDICTORS:
    term_pvalues = [
        pvalue
        for term, pvalue in full_model.pvalues.items()
        if predictor_for_term(term) == name
    ]
    if not term_pvalues or all(pvalue >= ALPHA for pvalue in term_pvalues):
        dropped.append(name)

kept = [name for name in PREDICTORS if name not in dropped]
print(
    "Dropped predictors (not significant at p < 0.05): "
    + (", ".join(dropped) if dropped else "none")
)
print()

trimmed_formula = "Sale_Price ~ " + " + ".join(kept) if kept else "Sale_Price ~ 1"
trimmed_model = smf.ols(trimmed_formula, data=train_df).fit()

print("Trimmed model summary")
print(trimmed_model.summary())
print()

print("Side-by-side comparison")
label_width = 28
value_width = 12
print(f"{'metric':<{label_width}} {'full':>{value_width}} {'trimmed':>{value_width}}")
print(
    f"{'adjusted R-squared':<{label_width}} "
    f"{full_model.rsquared_adj:>{value_width}.4f} "
    f"{trimmed_model.rsquared_adj:>{value_width}.4f}"
)
print(f"{'p-value':<{label_width}} {'full':>{value_width}} {'trimmed':>{value_width}}")

terms = list(dict.fromkeys([*full_model.pvalues.index, *trimmed_model.pvalues.index]))
for term in terms:
    full_p = (
        f"{full_model.pvalues[term]:.4f}" if term in full_model.pvalues.index else "n/a"
    )
    if term in trimmed_model.pvalues.index:
        trimmed_p = f"{trimmed_model.pvalues[term]:.4f}"
    else:
        trimmed_p = "dropped"
    print(f"{term:<{label_width}} {full_p:>{value_width}} {trimmed_p:>{value_width}}")
