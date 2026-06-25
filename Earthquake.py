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

    uploaded_file = st.file_uploader("Upload Earthquake Data CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        required_columns = [
            "Origin Time",
            "Latitude",
            "Longitude",
            "Depth",
            "Magnitude",
            "Location"
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
            return

        df["Origin Time"] = (
            df["Origin Time"]
            .astype(str)
            .str.replace(" IST", "", regex=False)
            .str.replace(" UTC", "", regex=False)
        )

        df["Origin Time"] = pd.to_datetime(
            df["Origin Time"],
            errors="coerce"
        )

        df = df.dropna(subset=[
            "Origin Time",
            "Latitude",
            "Longitude",
            "Depth",
            "Magnitude",
            "Location"
        ])

        if len(df) < 2:
            st.error("No valid earthquake records found after preprocessing.")
            st.stop()

        df["Timestamp"] = df["Origin Time"].astype("int64") // 10**9

        df["Location"] = df["Location"].astype(str)

        locations = sorted(df["Location"].unique())

        X = df[["Latitude", "Longitude", "Depth", "Timestamp"]]
        y = df["Magnitude"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        st.sidebar.subheader("📌 Predict Future Earthquake Magnitudes")

        selected_location = st.sidebar.selectbox(
            "📍 Select Location",
            locations
        )

        future_year = st.sidebar.slider(
            "📅 Select Future Year",
            2025,
            2100,
            2030
        )

        location_data = df[df["Location"] == selected_location]

        if location_data.empty:
            st.error("No data available for this location.")
            return

        avg_latitude = location_data["Latitude"].mean()
        avg_longitude = location_data["Longitude"].mean()
        avg_depth = location_data["Depth"].mean()

        future_data = pd.DataFrame({
            "Latitude": [avg_latitude],
            "Longitude": [avg_longitude],
            "Depth": [avg_depth],
            "Timestamp": [datetime(future_year, 1, 1).timestamp()]
        })

        future_prediction = model.predict(future_data)[0]

        st.markdown(
            f"""
            <div style="padding:12px;background:#006400;color:white;
            border-radius:10px;text-align:center;font-size:20px;">
            📊 Predicted Magnitude for <b>{selected_location}</b> in
            <b>{future_year}</b> : <b>{future_prediction:.2f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📈 Magnitude Distribution")

            x = np.linspace(
                future_prediction - 1,
                future_prediction + 1,
                100
            )

            y_plot = np.exp(
                -0.5 * ((x - future_prediction) / 0.3) ** 2
            ) / (0.3 * np.sqrt(2 * np.pi))

            fig1 = go.Figure()

            fig1.add_trace(
                go.Scatter(
                    x=x,
                    y=y_plot,
                    mode="lines"
                )
            )

            fig1.add_vline(
                x=future_prediction,
                line_dash="dash"
            )

            fig1.update_layout(
                xaxis_title="Magnitude",
                yaxis_title="Probability Density"
            )

            st.plotly_chart(fig1, use_container_width=True)

        with col2:

            st.subheader("📈 Future Trend")

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=[future_year],
                    y=[future_prediction],
                    mode="markers"
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=[
                        future_year - 5,
                        future_year,
                        future_year + 5
                    ],
                    y=[
                        future_prediction - 0.2,
                        future_prediction,
                        future_prediction + 0.2
                    ],
                    mode="lines"
                )
            )

            fig2.update_layout(
                xaxis_title="Year",
                yaxis_title="Magnitude"
            )

            st.plotly_chart(fig2, use_container_width=True)

        st.sidebar.subheader("📈 Model Performance")

        y_pred = model.predict(X_test)

        st.sidebar.write(
            f"🔹 MAE : {mean_absolute_error(y_test, y_pred):.2f}"
        )

        st.sidebar.write(
            f"🔹 R² Score : {r2_score(y_test, y_pred):.2f}"
        )

        st.sidebar.success("🌟 Enjoy Exploring Earthquake Trends!")

    else:
        st.warning("Please upload the earthquake dataset to proceed.")


if __name__ == "__main__":
    earthquake_prediction()
