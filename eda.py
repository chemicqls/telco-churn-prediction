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
df.info()

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

# -----------------------------------------------------------------
# 8. Does monthly bill affect churn?
#    Hypothesis: higher bills -> more churn (cost sensitivity).
# -----------------------------------------------------------------
# Mask: True for customers paying MORE than $70/month.
mask_high = df["MonthlyCharges"] > 70

# Mask: True for customers paying $70/month or LESS.
mask_low = df["MonthlyCharges"] <= 70

# Filter df down to each group, same df[mask] pattern as before.
high_bill = df[mask_high]
low_bill = df[mask_low]

# len() counts the rows in each filtered table.
print("\nMonthly charges > $70:", len(high_bill))
print(high_bill["Churn"].value_counts(normalize=True))

print("\nMonthly charges <= $70:", len(low_bill))
print(low_bill["Churn"].value_counts(normalize=True))

# -----------------------------------------------------------------
# 9. Does internet service type affect churn?
#    Hypothesis: No internet < DSL < Fiber optic, because fiber is the
#    most expensive plan.
# -----------------------------------------------------------------
# groupby("InternetService") splits the table into one group per value
# in that column (DSL, Fiber optic, No), then value_counts(normalize=True)
# gives the Yes/No churn percentages inside each group.
churn_by_internet = df.groupby("InternetService")["Churn"].value_counts(normalize=True)
print("\nChurn rate by internet service:")
print(churn_by_internet)

# Group sizes, so you can sanity-check they add up to 7,043.
print("\nCustomers per internet service type:")
print(df["InternetService"].value_counts())

# -----------------------------------------------------------------
# 10. Does monthly bill still matter among Fiber optic customers only?
#     Hypothesis: yes, the gap stays large, because price pushes people
#     out even on the premium plan.
# -----------------------------------------------------------------
# Mask: True only where BOTH conditions are True (& = "and").
# Each condition needs its own parentheses.
mask_fiber_high = (df["InternetService"] == "Fiber optic") & (df["MonthlyCharges"] > 70)
mask_fiber_low = (df["InternetService"] == "Fiber optic") & (df["MonthlyCharges"] <= 70)

# Filter df down to each group, same df[mask] pattern as before.
fiber_high = df[mask_fiber_high]
fiber_low = df[mask_fiber_low]

# Group sizes first: a churn percentage from a tiny group is unreliable.
print("\nFiber optic, monthly charges > $70:", len(fiber_high))
print(fiber_high["Churn"].value_counts(normalize=True))

print("\nFiber optic, monthly charges <= $70:", len(fiber_low))
print(fiber_low["Churn"].value_counts(normalize=True))

# -----------------------------------------------------------------
# 11. One-hot encoding test (single column first)
# -----------------------------------------------------------------
# pd.get_dummies() returns a NEW table; df itself is unchanged.
# columns=["Contract"]  -> only convert this text column
# drop_first=True       -> drop one category so the flags aren't redundant
df_encoded = pd.get_dummies(df, columns=["Contract"], drop_first=True)

# Show only the columns that changed, so the output is easy to read.
print("\nColumns after encoding Contract:")
print(df_encoded.columns.tolist())
print(df_encoded[["Contract_One year", "Contract_Two year"]].head())

# -----------------------------------------------------------------
# 12. One-hot encode three text columns at once
# -----------------------------------------------------------------
df_encoded = pd.get_dummies(df, columns=["Contract", "InternetService", "PaymentMethod"], drop_first=True)

# .shape returns (rows, columns). We expect 7043 rows and 25 columns.
print("\nShape after encoding:", df_encoded.shape)
print(df_encoded.columns.tolist())

# -----------------------------------------------------------------
# 13. One-hot encode the remaining text columns
# -----------------------------------------------------------------
# Built from df_encoded (not df), so the Contract / InternetService /
# PaymentMethod encoding from section 12 is kept.
text_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling",
]

df_encoded2 = pd.get_dummies(df_encoded, columns=text_columns, drop_first=True)

# Expected shape: (7043, 32)
print("\nShape after encoding everything:", df_encoded2.shape)
print(df_encoded2.columns.tolist())