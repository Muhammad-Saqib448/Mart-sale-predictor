import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt

# -------------------------------
# Load model and encoders
# -------------------------------
model = joblib.load("bigmart_sales_model.pkl")

encoder_files = {
    'Item_Identifier': 'encoders/Item_Identifier.pkl',
    'Item_Fat_Content': 'encoders/Item_Fat_Content.pkl',
    'Item_Type': 'encoders/Item_Type.pkl',
    'Outlet_Identifier': 'encoders/Outlet_Identifier.pkl',
    'Outlet_Size': 'encoders/Outlet_Size.pkl',
    'Outlet_Location_Type': 'encoders/Outlet_Location_Type.pkl',
    'Outlet_Type': 'encoders/Outlet_Type.pkl',
    'Item_Category': 'encoders/Item_Category.pkl'
}

encoders = {col: joblib.load(path) for col, path in encoder_files.items()}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Big Mart Sales Predictor", layout="wide")
st.title("🛒 Big Mart Sales Prediction Dashboard")
st.write("Select an item and outlet to explore predicted sales with varying MRP and Visibility.")

# -------------------------------
# Sidebar Inputs (Dynamic)
# -------------------------------
st.sidebar.header("Select Item & Outlet")

Item_Identifier = st.sidebar.selectbox("Item Identifier", encoders['Item_Identifier'].classes_)
Outlet_Identifier = st.sidebar.selectbox("Outlet Identifier", encoders['Outlet_Identifier'].classes_)

Item_Fat_Content = st.sidebar.selectbox("Item Fat Content", encoders['Item_Fat_Content'].classes_)
Item_Type = st.sidebar.selectbox("Item Type", encoders['Item_Type'].classes_)
Outlet_Size = st.sidebar.selectbox("Outlet Size", encoders['Outlet_Size'].classes_)
Outlet_Location_Type = st.sidebar.selectbox("Outlet Location Type", encoders['Outlet_Location_Type'].classes_)
Outlet_Type = st.sidebar.selectbox("Outlet Type", encoders['Outlet_Type'].classes_)
Item_Category = st.sidebar.selectbox("Item Category", encoders['Item_Category'].classes_)

Item_Weight = st.sidebar.number_input("Item Weight", min_value=0.0, value=12.0, step=0.1)
Outlet_Establishment_Year = st.sidebar.number_input("Outlet Establishment Year", min_value=1950, max_value=2025, value=1999)

# -------------------------------
# Main Panel Inputs for Chart
# -------------------------------
st.subheader("Explore Predicted Sales vs MRP and Visibility")

Item_MRP = st.slider("Item MRP Range", min_value=0.0, max_value=1000.0, value=(100.0, 300.0), step=1.0)
Item_Visibility = st.slider("Item Visibility Range", min_value=0.0, max_value=0.5, value=(0.01, 0.1), step=0.01)

# -------------------------------
# Feature Engineering Helper
# -------------------------------
def encode(col, val):
    return encoders[col].transform([val])[0]

Outlet_Age = 2025 - Outlet_Establishment_Year

# -------------------------------
# Generate Prediction Grid
# -------------------------------
mrp_values = np.linspace(Item_MRP[0], Item_MRP[1], 20)
visibility_values = np.linspace(Item_Visibility[0], Item_Visibility[1], 20)

data_chart = []

for mrp in mrp_values:
    for vis in visibility_values:
        Price_per_Weight = mrp / Item_Weight if Item_Weight > 0 else 0
        Visibility_Ratio = vis / Item_Weight if Item_Weight > 0 else 0

        features = np.array([[
            encode('Item_Identifier', Item_Identifier),
            Item_Weight,
            encode('Item_Fat_Content', Item_Fat_Content),
            vis,
            encode('Item_Type', Item_Type),
            mrp,
            encode('Outlet_Identifier', Outlet_Identifier),
            Outlet_Establishment_Year,
            encode('Outlet_Size', Outlet_Size),
            encode('Outlet_Location_Type', Outlet_Location_Type),
            encode('Outlet_Type', Outlet_Type),
            encode('Item_Category', Item_Category),
            Price_per_Weight,
            Visibility_Ratio,
            Outlet_Age
        ]])

        pred_sales = model.predict(features)[0]
        data_chart.append([mrp, vis, pred_sales])

df_chart = pd.DataFrame(data_chart, columns=['MRP', 'Visibility', 'Predicted_Sales'])

# -------------------------------
# Interactive Altair Chart
# -------------------------------
chart = alt.Chart(df_chart).mark_circle(size=60).encode(
    x='MRP',
    y='Visibility',
    color='Predicted_Sales',
    tooltip=['MRP', 'Visibility', 'Predicted_Sales']
).interactive()

st.altair_chart(chart, use_container_width=True)

st.info("Adjust the sliders and dropdowns to explore how sales predictions vary with MRP and Visibility.")
