import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Rainfall.csv")

# Debugging: Print the column names to check if 'STATE' exists
st.write("Columns in the dataset:", data.columns.tolist())

# Check if 'STATE' is in the columns
if 'STATE' not in data.columns:
    st.error("Column 'STATE' not found in dataset!")
    st.stop()  # Stop further execution if the column is missing

# Check for missing values and handle them (e.g., replace with the mean of each column)
imputer = SimpleImputer(strategy='mean')
data.iloc[:, 2:] = imputer.fit_transform(data.iloc[:, 2:])  # Apply imputation to numeric columns

# Include "YEAR" as a feature
X = data[["YEAR"]].join(data.iloc[:, 2:-5])  # Combine "YEAR" with monthly/seasonal rainfall
y = data["ANNUAL"].values  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Define months for dropdown (ensure this list matches the number of months in the data)
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Streamlit app
st.title("Rainfall Prediction and Flood Risk Analysis")
st.sidebar.header("User Input")

# Dropdown menu for selecting a state
state = st.sidebar.selectbox("Select State", data["STATE"].unique())

# Dropdown menu for selecting a month
month = st.sidebar.selectbox("Select Month", months)

# Text input for future year
future_year = st.sidebar.number_input("Enter Future Year (e.g., 2025)", min_value=2024, step=1)

# Prediction button
if st.sidebar.button("Predict Flood Risk"):
    # Debugging: Verify selected state
    st.write(f"Selected state: {state}")

    # Extract the relevant data for the selected month
    # Adjust slicing to ensure all months are included
    monthly_columns = data.columns[2:14]  # Assuming monthly data is in columns 2 to 13
    st.write("Monthly columns selected:", monthly_columns)
    
    # Select the appropriate monthly data for the selected state
    selected_month_data = data[data["STATE"] == state][monthly_columns].iloc[:, months.index(month)]

    # Debugging: Verify the selected month's data
    st.write(f"Selected month's data for {month}: {selected_month_data}")

    # Get the average rainfall for the selected month
    month_avg_rainfall = selected_month_data.mean()

    # Prepare input data: Use the future year and average rainfall for prediction
    input_data = [future_year] + [month_avg_rainfall] * (X.shape[1] - 1)  # Matching the number of features in X
    input_data_array = np.array(input_data).reshape(1, -1)
    
    # Scale the input data (same scaling as the training data)
    input_data_scaled = scaler.transform(input_data_array)
    
    # Predict the annual rainfall
    predicted_annual_rainfall = model.predict(input_data_scaled)[0]
    flood_risk_threshold = 1500  # Example threshold for flood risk
    flood_risk = "No flood risk" if predicted_annual_rainfall < flood_risk_threshold else "Flood risk detected"
    
    # Display results
    st.subheader(f"Prediction for {state} in {month}, {future_year}")
    st.write(f"Predicted Annual Rainfall: {predicted_annual_rainfall:.2f} mm")
    st.write(f"Flood Risk: **{flood_risk}**")
    
    # Generate data for visualization: Average rainfall for each month
    monthly_rainfall = data[data["STATE"] == state][monthly_columns].mean(axis=0)
    
    # Calculate flood risk for each month
    flood_risk_monthly = ["High" if rainfall > flood_risk_threshold / 12 else "Low" for rainfall in monthly_rainfall]
    
    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Line Graph for Monthly Rainfall
    ax[0].plot(months, monthly_rainfall, marker='o', label="Average Rainfall")
    ax[0].axhline(flood_risk_threshold / 12, color="red", linestyle="--", label="Flood Risk Threshold")
    ax[0].set_title(f"Average Monthly Rainfall in {state}")
    ax[0].set_ylabel("Rainfall (mm)")
    ax[0].legend()
    
    # Bar Graph for Flood Risk in Months
    ax[1].bar(months, monthly_rainfall, color=['red' if risk == "High" else 'green' for risk in flood_risk_monthly])
    ax[1].set_title(f"Flood Risk by Month in {state}")
    ax[1].set_ylabel("Rainfall (mm)")
    
    # Show plots in Streamlit
    st.pyplot(fig)
