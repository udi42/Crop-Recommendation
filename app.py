import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

st.title("Crop Recommendation Assistant")

# Load the Excel data
excel = pd.read_excel('Crop.xlsx', header=0)

# Create a sidebar for user input
st.sidebar.header("Please enter the following details:")
nitrogen_content = st.sidebar.number_input("Enter ratio of Nitrogen in the soil:", min_value=0.0)
phosphorus_content = st.sidebar.number_input("Enter ratio of Phosphorus in the soil:", min_value=0.0)
potassium_content = st.sidebar.number_input("Enter ratio of Potassium in the soil:", min_value=0.0)
temperature_content = st.sidebar.number_input("Enter average Temperature value around the field (*C):", min_value=0.0)
humidity_content = st.sidebar.number_input("Enter average percentage of Humidity around the field (%):", min_value=0.0)
ph_content = st.sidebar.number_input("Enter PH value of the soil:", min_value=0.0)
rainfall = st.sidebar.number_input("Enter average amount of Rainfall around the field (mm):", min_value=0.0)

# Encode the crop labels
le = preprocessing.LabelEncoder()
crop = le.fit_transform(list(excel["CROP"]))

# Create features
NITROGEN = list(excel["NITROGEN"])
PHOSPHORUS = list(excel["PHOSPHORUS"])
POTASSIUM = list(excel["POTASSIUM"])
TEMPERATURE = list(excel["TEMPERATURE"])
HUMIDITY = list(excel["HUMIDITY"])
PH = list(excel["PH"])
RAINFALL = list(excel["RAINFALL"])

features = np.array([NITROGEN, PHOSPHORUS, POTASSIUM, TEMPERATURE, HUMIDITY, PH, RAINFALL])
features = features.transpose()

# Train the KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features, crop)

# Function to recommend crops
def recommend_crops(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    user_input = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall])
    user_input = user_input.reshape(1, -1)
    prediction = model.predict(user_input)
    crop_name = le.inverse_transform(prediction)[0]
    return crop_name

# Add a "Recommend Crops" button
if st.sidebar.button("Recommend Crops"):
    recommended_crop = recommend_crops(nitrogen_content, phosphorus_content, potassium_content, temperature_content, humidity_content, ph_content, rainfall)
    st.write("The best crop that you can grow is:")
    st.markdown(f"<span style='color:red;font-size:18px'>{recommended_crop}</span>", unsafe_allow_html=True)

# Additional information
st.write("Additional Information:")
st.write("Nitrogen Level:", nitrogen_content)
st.write("Phosphorus Level:", phosphorus_content)
st.write("Potassium Level:", potassium_content)
st.write("Temperature Level:", temperature_content)
st.write("Humidity Level:", humidity_content)
st.write("PH Level:", ph_content)
st.write("Rainfall Level:", rainfall)
