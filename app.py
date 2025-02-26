import streamlit as st
from Earthquake import earthquake_prediction
from Rainfall_prediction_app import rainfall_prediction

# Configure Streamlit page settings
st.set_page_config(page_title="Disaster Prediction System", layout="wide")

# Title and introduction
st.title("🌎 Disaster Prediction System")
st.write("Predict the likelihood of **Floods** and **Earthquakes** based on historical data and machine learning models.")

# Sidebar for user selection
st.sidebar.title("🔍 Select Disaster Type")
option = st.sidebar.radio("Choose a prediction model:", ["Flood Prediction", "Earthquake Prediction"])

# Display a relevant image based on the selected disaster type
if option == "Flood Prediction":
    st.image("https://media.giphy.com/media/3ohhwytHcusSCXXOUg/giphy.gif", caption="Flood Analysis", width=300)
    st.success("🔄 Loading Flood Prediction Model...")
    rainfall_prediction()  # Calls the Flood prediction function

elif option == "Earthquake Prediction":
    st.image("https://media.giphy.com/media/3o7aCSPqXE5C6T8tBC/giphy.gif", caption="Earthquake Alert!", width=300)
    st.success("🔄 Loading Earthquake Prediction Model...")
    earthquake_prediction()  # Calls the Earthquake prediction function

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **Developed for real-time disaster risk analysis!**")
