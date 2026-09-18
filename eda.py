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

# -----------------------------------------------------------------
# 6. Does tenure affect churn? Testing the hypothesis: newer customers
#    churn more because long-time customers have built up loyalty.
# -----------------------------------------------------------------
# Step 1: build the mask -- True for customers who've been here 12 months or less
mask_new = df["tenure"] <= 12

# Step 2: filter df down to just those customers
new_customers = df[mask_new]

print("\nCustomers with tenure <= 12 months:", len(new_customers))
print(new_customers["Churn"].value_counts(normalize=True))

# Now the opposite group: customers here MORE than 12 months
mask_long = df["tenure"] > 12
long_customers = df[mask_long]

print("\nCustomers with tenure > 12 months:", len(long_customers))
print(long_customers["Churn"].value_counts(normalize=True))


# -----------------------------------------------------------------
# 7. Does tenure affect churn? 4-bucket breakdown
# -----------------------------------------------------------------
mask_bucket1 = df["tenure"] <= 12
mask_bucket2 = (df["tenure"] > 12) & (df["tenure"] <= 24)
mask_bucket3 = (df["tenure"] > 24) & (df["tenure"] <= 48)
mask_bucket4 = df["tenure"] >= 49

bucket1 = df[mask_bucket1]
bucket2 = df[mask_bucket2]
bucket3 = df[mask_bucket3]
bucket4 = df[mask_bucket4]

print("\nTenure 0-12 months:", len(bucket1))
print(bucket1["Churn"].value_counts(normalize=True))

print("\nTenure 13-24 months:", len(bucket2))
print(bucket2["Churn"].value_counts(normalize=True))

print("\nTenure 25-48 months:", len(bucket3))
print(bucket3["Churn"].value_counts(normalize=True))

print("\nTenure 49+ months:", len(bucket4))
print(bucket4["Churn"].value_counts(normalize=True))