# ============================================================
# Simple Linear Regression: Age vs Insurance Premium
# Using Entire Dataset (Recommended for Small Datasets)
# ============================================================

import sys

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# ── 1. Load Dataset ──────────────────────────────────────────

df = pd.read_csv("simplelinearregression.csv")

print("Dataset:")
print(df)

print("\nShape:", df.shape)


# ── 2. Features and Target ──────────────────────────────────

X = df[['Age']]
y = df['Premium']


# ── 3. Train Model on Entire Dataset ───────────────────────

model = LinearRegression()

model.fit(X, y)


# ── 4. Model Parameters ────────────────────────────────────

print("\n── Model Parameters ──")

print(f"Intercept (β₀): {model.intercept_:.4f}")

print(f"Coefficient (β₁): {model.coef_[0]:.4f}")

print(f"\nEquation:")

print(f"Premium = {model.intercept_:.2f} + ({model.coef_[0]:.2f}) × Age")


# ── 5. Predictions ─────────────────────────────────────────

y_pred = model.predict(X)


# ── 6. Evaluation Metrics ──────────────────────────────────

r2 = r2_score(y, y_pred)

mae = mean_absolute_error(y, y_pred)

mse = mean_squared_error(y, y_pred)

rmse = np.sqrt(mse)


print("\n── Evaluation Metrics ──")

print(f"R² Score : {r2:.4f}")

print(f"MAE      : {mae:.2f}")

print(f"MSE      : {mse:.2f}")

print(f"RMSE     : {rmse:.2f}")


# ── 7. Predict Premium for New Age ─────────────────────────

new_age = pd.DataFrame({'Age':[35]})

predicted_premium = model.predict(new_age)

print(f"\nPredicted Premium for Age 35 : {predicted_premium[0]:,.2f}")


# ── 8. Visualization ───────────────────────────────────────

fig, axes = plt.subplots(1,3,figsize=(18,5))

fig.suptitle(
    "Simple Linear Regression — Age vs Premium",
    fontsize=14,
    fontweight='bold'
)


# -----------------------------------------------------------
# Plot 1 : Regression Line
# -----------------------------------------------------------

ax1 = axes[0]

ax1.scatter(X,
            y,
            color='blue',
            s=80,
            label='Actual Data')


x_line = np.linspace(
            X['Age'].min(),
            X['Age'].max(),
            100
         ).reshape(-1,1)


ax1.plot(
        x_line,
        model.predict(
            pd.DataFrame(
                x_line,
                columns=['Age']
            )
        ),
        color='red',
        linewidth=2,
        label='Regression Line'
)


ax1.set_xlabel("Age")

ax1.set_ylabel("Premium")

ax1.set_title("Regression Line")

ax1.legend()

ax1.grid(True)



# -----------------------------------------------------------
# Plot 2 : Actual vs Predicted
# -----------------------------------------------------------

ax2 = axes[1]


ax2.scatter(
        y,
        y_pred,
        color='green',
        s=80
)


lims = [

min(y.min(), y_pred.min())-1000,

max(y.max(), y_pred.max())+1000

]


ax2.plot(
        lims,
        lims,
        'r--'
)


ax2.set_xlabel("Actual Premium")

ax2.set_ylabel("Predicted Premium")

ax2.set_title("Actual vs Predicted")

ax2.grid(True)



# -----------------------------------------------------------
# Plot 3 : Residual Plot
# -----------------------------------------------------------

ax3 = axes[2]


residuals = y - y_pred


ax3.scatter(
        y_pred,
        residuals,
        color='orange',
        s=80
)


ax3.axhline(
            y=0,
            color='red',
            linestyle='--'
)


ax3.set_xlabel("Predicted Premium")

ax3.set_ylabel("Residuals")

ax3.set_title("Residual Plot")

ax3.grid(True)


plt.tight_layout()

plt.show()
