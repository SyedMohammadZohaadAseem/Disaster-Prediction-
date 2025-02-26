import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import plotly.graph_objects as go

def earthquake_prediction():
    st.title("🌍 Earthquake Magnitude Prediction Dashboard")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload Earthquake Data CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Convert 'Origin Time' to datetime format
        if 'Origin Time' not in df.columns or 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            st.error("Invalid dataset! Ensure it contains 'Origin Time', 'Latitude', 'Longitude', 'Depth', and 'Magnitude'.")
            return

        df['Origin Time'] = pd.to_datetime(df['Origin Time'], errors='coerce')
        df = df.dropna()  # Remove NaT values
        df['Timestamp'] = df['Origin Time'].apply(lambda x: x.timestamp())

        # Extract unique locations for dropdown
        df['Location'] = df['Location'].astype(str)
        locations = df['Location'].unique().tolist()

        # Selecting features and target
        X = df[['Latitude', 'Longitude', 'Depth', 'Timestamp']]
        y = df['Magnitude']

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Sidebar inputs
        st.sidebar.subheader("📌 Predict Future Earthquake Magnitudes")
        selected_location = st.sidebar.selectbox("📍 Select Location", locations)
        future_year = st.sidebar.slider("📅 Select Future Year", min_value=2025, max_value=2100, value=2030)

        # Filter dataset for the selected location
        location_data = df[df['Location'] == selected_location]
        if location_data.empty:
            st.error("No data available for the selected location.")
        else:
            avg_latitude = location_data['Latitude'].mean()
            avg_longitude = location_data['Longitude'].mean()
            avg_depth = location_data['Depth'].mean()
            
            future_timestamps = np.array([datetime(future_year, 1, 1).timestamp()])
            future_data = pd.DataFrame({
                'Latitude': [avg_latitude],
                'Longitude': [avg_longitude],
                'Depth': [avg_depth],
                'Timestamp': future_timestamps
            })

            future_predictions = model.predict(future_data)

            st.markdown(f"""
                <div style="padding: 10px; background-color: #006400; color: white; border-radius: 10px; font-size: 18px; text-align: center;">
                📊 Predicted Magnitude for {selected_location} in {future_year}: <b>{future_predictions[0]:.2f}</b>
                </div>
            """, unsafe_allow_html=True)

            # Responsive Layout for Graphs
            with st.container():
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Predicted Magnitude Distribution")
                    x_values = np.linspace(future_predictions[0] - 1, future_predictions[0] + 1, 100)
                    y_values = np.exp(-0.5 * ((x_values - future_predictions[0]) / 0.3) ** 2) / (0.3 * np.sqrt(2 * np.pi))

                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Magnitude Distribution', line=dict(color='blue')))
                    fig1.add_vline(x=future_predictions[0], line=dict(color='red', dash='dash'), name='Predicted Magnitude')
                    fig1.update_layout(
                        title='Predicted Earthquake Magnitude Distribution',
                        xaxis_title='Magnitude',
                        yaxis_title='Probability Density',
                        width=600, height=350
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.subheader("📈 Earthquake Magnitude Trend Over Time")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=[future_year], y=future_predictions, mode='markers', marker=dict(color='red', size=8), name='Predicted Magnitude'))
                    fig2.add_trace(go.Scatter(x=[future_year - 5, future_year, future_year + 5],
                                              y=[future_predictions[0] - 0.2, future_predictions[0], future_predictions[0] + 0.2],
                                              mode='lines', line=dict(dash='dash', color='blue'), name='Trend'))
                    fig2.update_layout(
                        title='Predicted Earthquake Magnitude Over Future Years',
                        xaxis_title='Year',
                        yaxis_title='Predicted Magnitude',
                        width=600, height=350
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Model Performance Metrics
            st.sidebar.subheader("📈 Model Performance")
            st.sidebar.write(f"🔹 MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
            st.sidebar.write(f"🔹 R2 Score: {r2_score(y_test, model.predict(X_test)):.2f}")

            st.sidebar.success("🌟 Enjoy Exploring Earthquake Trends!")
    else:
        st.warning("Please upload the earthquake dataset to proceed.")

# Ensure the function only runs when called
if __name__ == "__main__":
    earthquake_prediction()
