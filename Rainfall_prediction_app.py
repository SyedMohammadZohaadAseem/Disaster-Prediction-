import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def rainfall_prediction():
    st.title("🌧️ Rainfall Prediction and Flood Risk Analysis")
    
    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload Rainfall Data CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Check if 'STATE' column exists
        if 'STATE' not in data.columns:
            st.error("Column 'STATE' not found in dataset! Please upload a valid file.")
            return

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        data.iloc[:, 2:] = imputer.fit_transform(data.iloc[:, 2:])

        # Include "YEAR" as a feature
        X = data[["YEAR"]].join(data.iloc[:, 2:-5])  
        y = data["ANNUAL"].values  

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Define months
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

        # Sidebar input
        st.sidebar.header("User Input")
        state = st.sidebar.selectbox("Select State", data["STATE"].unique())
        month = st.sidebar.selectbox("Select Month", months)
        future_year = st.sidebar.number_input("Enter Future Year (e.g., 2025)", min_value=2024, step=1)

        # Predict button
        if st.sidebar.button("Predict Flood Risk"):
            # Extract monthly data
            monthly_columns = data.columns[2:14]  
            selected_month_data = data[data["STATE"] == state][monthly_columns].iloc[:, months.index(month)]

            # Get average rainfall
            month_avg_rainfall = selected_month_data.mean()

            # Prepare input data
            input_data = [future_year] + [month_avg_rainfall] * (X.shape[1] - 1)  
            input_data_array = np.array(input_data).reshape(1, -1)
            input_data_scaled = scaler.transform(input_data_array)

            # Predict
            predicted_annual_rainfall = model.predict(input_data_scaled)[0]
            flood_risk_threshold = 1500  
            flood_risk = "No flood risk" if predicted_annual_rainfall < flood_risk_threshold else "Flood risk detected"

            # Display results
            st.subheader(f"Prediction for {state} in {month}, {future_year}")
            st.write(f"**Predicted Annual Rainfall:** {predicted_annual_rainfall:.2f} mm")
            st.write(f"**Flood Risk:** **{flood_risk}**")

            # Visualization
            monthly_rainfall = data[data["STATE"] == state].iloc[:, 2:-5].mean(axis=0)
            months = months[:len(monthly_rainfall)]
            flood_risk_monthly = ["High" if rainfall > flood_risk_threshold / 12 else "Low" for rainfall in monthly_rainfall]

            fig, ax = plt.subplots(2, 1, figsize=(10, 8))

            # Line Graph
            ax[0].plot(months, monthly_rainfall, marker='o', label="Average Rainfall")
            ax[0].axhline(flood_risk_threshold / 12, color="red", linestyle="--", label="Flood Risk Threshold")
            ax[0].set_title(f"Average Monthly Rainfall in {state}")
            ax[0].set_ylabel("Rainfall (mm)")
            ax[0].legend()

            # Bar Graph
            ax[1].bar(months, monthly_rainfall, color=['red' if risk == "High" else 'green' for risk in flood_risk_monthly])
            ax[1].set_title(f"Flood Risk by Month in {state}")
            ax[1].set_ylabel("Rainfall (mm)")

            st.pyplot(fig)

# Ensure the function only runs when called
if __name__ == "__main__":
    rainfall_prediction()
