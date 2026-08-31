"""
Step 1: Data Loading & Cleaning
--------------------------------
Goal: Load the raw Telco Customer Churn CSV, inspect it, and fix a known
data quality issue in the TotalCharges column before we do anything else.
"""

import pandas as pd

# -----------------------------------------------------------------
# 1. Load the raw CSV
# -----------------------------------------------------------------
# pd.read_csv() reads a CSV file and returns a DataFrame -- pandas'
# core data structure, basically a spreadsheet/table you can manipulate
# in code. Since the CSV is in the same folder as this script, we can
# just use its filename directly (no full path needed).
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -----------------------------------------------------------------
# 2. First look at the data
# -----------------------------------------------------------------
# .shape returns (rows, columns) -- a quick sense of dataset size.
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# .head() shows the first 5 rows -- a quick visual sanity check that
# the data loaded correctly and looks like what you expect.
print("\nFirst 5 rows:")
print(df.head())

# .info() shows every column's name, how many non-null values it has,
# and its data type (dtype). This is the single most useful command
# for spotting problems early -- like a numeric column that's secretly
# stored as text (object), which is exactly the issue we're about to fix.
print("\nColumn info:")
print(df.info())

# -----------------------------------------------------------------
# 3. Investigate the known TotalCharges issue
# -----------------------------------------------------------------
# TotalCharges SHOULD be a numeric column (it's a dollar amount), but
# if you look at .info() above, you'll likely see it's dtype "object"
# (pandas' way of saying "text/string"), not float64. That's a red flag.
#
# Why does this happen? Some rows have TotalCharges as a blank string
# (" ") instead of an actual empty/missing value, which forces the
# ENTIRE column to be stored as text instead of numbers.
#
# Let's find exactly which rows have this problem:
blank_mask = df["TotalCharges"] == " "
# This creates a "boolean mask" -- a True/False value for every row,
# True wherever TotalCharges is exactly a blank space string.

print(f"\nNumber of rows with blank TotalCharges: {blank_mask.sum()}")
# .sum() on a boolean Series counts the True values (True=1, False=0).

# Let's actually look at those problem rows to understand them:
print("\nRows with blank TotalCharges:")
print(df[blank_mask][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])
# df[blank_mask] filters the DataFrame down to ONLY the rows where
# blank_mask is True. Then [[...]] selects just a few relevant columns
# so the output isn't overwhelming.

# -----------------------------------------------------------------
# 4. Fix the TotalCharges column
# -----------------------------------------------------------------
# pd.to_numeric() converts a column to actual numbers. The parameter
# errors="coerce" tells pandas: "if you find a value you can't convert
# to a number (like our blank strings), don't crash -- just turn it
# into NaN (pandas' 'missing value' marker) instead."
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Now let's confirm the fix worked and see how many NaNs we introduced.
print(f"\nMissing TotalCharges after conversion: {df['TotalCharges'].isnull().sum()}")

# Looking at the problem rows above, notice they all have tenure == 0,
# meaning these are brand new customers who haven't been billed a full
# cycle yet -- that's WHY TotalCharges is blank, not random bad data.
# Since TotalCharges should logically be 0 for a customer with 0 tenure,
# we fill those specific NaNs with 0 rather than guessing or dropping them.
df["TotalCharges"] = df["TotalCharges"].fillna(0)
# fillna() replaces NaN values with whatever you pass in -- here, 0.

# Final check: confirm no missing values remain in this column.
print(f"Missing TotalCharges after fillna: {df['TotalCharges'].isnull().sum()}")
print(f"TotalCharges dtype is now: {df['TotalCharges'].dtype}")

# -----------------------------------------------------------------
# 5. Quick check on the target variable (Churn)
# -----------------------------------------------------------------
# .value_counts() counts how many times each unique value appears --
# essential for understanding class balance before we ever train a model.
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print("\nChurn distribution (as percentages):")
print(df["Churn"].value_counts(normalize=True) * 100)
# normalize=True converts raw counts into proportions (0 to 1) instead
# of counts -- multiplying by 100 turns that into a percentage.