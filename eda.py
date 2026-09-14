# eda.py
# Step 2 of Project 1: Exploratory Data Analysis (EDA)
# Goal: explore whether Contract type affects churn rate, one contract type
# at a time, using the same mask + filter pattern from load_data.py.

import pandas as pd

# -----------------------------------------------------------------
# 1. Load and clean the data (same fix as load_data.py)
# -----------------------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# -----------------------------------------------------------------
# 2. Check churn rate for Month-to-month customers only
# -----------------------------------------------------------------
# Step 1: build the mask -- True for rows where Contract is "Month-to-month"
mask_mtm = df["Contract"] == "Month-to-month"

# Step 2: use the mask to filter df down to only those rows
month_to_month = df[mask_mtm]

print("Month-to-month customers:", len(month_to_month))
print(month_to_month["Churn"].value_counts(normalize=True))

# -----------------------------------------------------------------
# 3. Check churn rate for One year customers only
# -----------------------------------------------------------------
mask_one_year = df["Contract"] == "One year"
one_year = df[mask_one_year]

print("\nOne year customers:", len(one_year))
print(one_year["Churn"].value_counts(normalize=True))

# -----------------------------------------------------------------
# 4. Check churn rate for Two year customers only
# -----------------------------------------------------------------
mask_two_year = df["Contract"] == "Two year"
two_year = df[mask_two_year]

print("\nTwo year customers:", len(two_year))
print(two_year["Churn"].value_counts(normalize=True))

# -----------------------------------------------------------------
# 5. Same result as the three manual blocks above, but using groupby
# -----------------------------------------------------------------
print("\nSame result using groupby:")
churn_by_contract = df.groupby("Contract")["Churn"].value_counts(normalize=True)
print(churn_by_contract)