# app.py — Fully user-friendly Streamlit Big Mart Sales Predictor
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt

st.set_page_config(page_title="Big Mart Sales Predictor", layout="wide")

# -------------------------
# Load model + encoders
# -------------------------
MODEL_PATH = "bigmart_sales_model.pkl"
ENCODER_PATHS = {
    'Item_Identifier': 'encoders/Item_Identifier.pkl',
    'Item_Fat_Content': 'encoders/Item_Fat_Content.pkl',
    'Item_Type': 'encoders/Item_Type.pkl',
    'Outlet_Identifier': 'encoders/Outlet_Identifier.pkl',
    'Outlet_Size': 'encoders/Outlet_Size.pkl',
    'Outlet_Location_Type': 'encoders/Outlet_Location_Type.pkl',
    'Outlet_Type': 'encoders/Outlet_Type.pkl',
    'Item_Category': 'encoders/Item_Category.pkl'
}

@st.cache_resource
def load_model_encoders(model_path, enc_paths):
    model = joblib.load(model_path)
    encs = {k: joblib.load(v) for k, v in enc_paths.items()}
    return model, encs

try:
    model, encoders = load_model_encoders(MODEL_PATH, ENCODER_PATHS)
except Exception as e:
    st.error(f"Could not load model/encoders: {e}")
    st.stop()

# -------------------------
# Helper: safe encode
# -------------------------
def safe_encode(col, val):
    """Encode a single value using saved LabelEncoder.
       If unseen, return -1."""
    le = encoders[col]
    try:
        if val in le.classes_:
            return int(le.transform([val])[0])
        else:
            return -1
    except Exception:
        return -1

# -------------------------
# Friendly mappings
# -------------------------
fat_options = ['Low Fat', 'Regular', 'Non-Edible']  # adjust to your dataset
outlet_size_options = ['Small', 'Medium', 'High']
outlet_location_options = ['Tier 1', 'Tier 2', 'Tier 3']
outlet_type_options = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']

# Item Category
category_classes = list(encoders['Item_Category'].classes_)
category_labels = []
for i, cls in enumerate(category_classes):
    if i == 0: label = "Foods"
    elif i == 1: label = "Non-Consumables"
    elif i == 2: label = "Drinks"
    else: label = str(cls)
    category_labels.append(label)
cat_label_to_class = {lbl: category_classes[i] for i, lbl in enumerate(category_labels)}

# -------------------------
# Sidebar: Inputs
# -------------------------
st.sidebar.header("Item & Outlet Details")

# Item Identifier (searchable dropdown)
item_ids = list(encoders['Item_Identifier'].classes_)
Item_Identifier = st.sidebar.selectbox("Item Identifier", item_ids)

# Item Fat Content
Item_Fat_Content = st.sidebar.selectbox("Item Fat Content", fat_options)

# Item Type
item_types = list(encoders['Item_Type'].classes_)
Item_Type = st.sidebar.selectbox("Item Type", item_types)

# Outlet Identifier
outlet_ids = list(encoders['Outlet_Identifier'].classes_)
Outlet_Identifier = st.sidebar.selectbox("Outlet Identifier", outlet_ids)

# Outlet Size, Location, Type
Outlet_Size = st.sidebar.selectbox("Outlet Size", outlet_size_options)
Outlet_Location_Type = st.sidebar.selectbox("Outlet Location Type", outlet_location_options)
Outlet_Type = st.sidebar.selectbox("Outlet Type", outlet_type_options)

# Item Category
Item_Category_label = st.sidebar.selectbox("Item Category", category_labels)

# Numerical inputs
st.sidebar.markdown("---")
Item_Weight = st.sidebar.number_input("Item Weight (kg)", min_value=0.01, value=12.0, step=0.1)
Item_Visibility = st.sidebar.number_input("Item Visibility (0-1)", min_value=0.0, value=0.05, step=0.01)
Item_MRP = st.sidebar.number_input("Item MRP", min_value=0.0, value=200.0, step=0.5)
Outlet_Establishment_Year = st.sidebar.number_input("Outlet Establishment Year", min_value=1950, max_value=2025, value=1999, step=1)

# Prediction options
st.sidebar.markdown("---")
show_chart = st.sidebar.checkbox("Show sensitivity chart (MRP vs Visibility)", value=False)
mrp_range_default = (max(1.0, Item_MRP * 0.7), Item_MRP * 1.3)
vis_range_default = (max(0.001, Item_Visibility * 0.7), Item_Visibility * 1.3)
mrp_min, mrp_max = st.sidebar.slider("MRP range for chart", 0.0, 2000.0, value=mrp_range_default, step=1.0)
vis_min, vis_max = st.sidebar.slider("Visibility range for chart", 0.0, 1.0, value=vis_range_default, step=0.01)

# -------------------------
# Main Dashboard
# -------------------------
st.markdown("<h1 style='text-align:center;'>🛒 Big Mart Sales Predictor</h1>", unsafe_allow_html=True)

# Input summary
st.markdown("### Input Summary")
col1, col2 = st.columns(2)
with col1:
    st.write("**Selected Item**")
    st.write(f"- Identifier: `{Item_Identifier}`")
    st.write(f"- Type: {Item_Type}")
    st.write(f"- Fat Content: {Item_Fat_Content}")
    st.write(f"- Category: {Item_Category_label}")
with col2:
    st.write("**Selected Outlet**")
    st.write(f"- Outlet ID: `{Outlet_Identifier}`")
    st.write(f"- Size: {Outlet_Size}")
    st.write(f"- Location Tier: {Outlet_Location_Type}")
    st.write(f"- Type: {Outlet_Type}")

st.write("---")

# -------------------------
# Predict button
# -------------------------
predict_col, info_col = st.columns([1,1])
with predict_col:
    if st.button("Predict Sales", type="primary"):
        # Map category label to class
        Item_Category_class = cat_label_to_class[Item_Category_label]

        # Encode all categorical variables
        enc_Item_Identifier = safe_encode('Item_Identifier', Item_Identifier)
        enc_Item_Fat_Content = safe_encode('Item_Fat_Content', Item_Fat_Content)
        enc_Item_Type = safe_encode('Item_Type', Item_Type)
        enc_Outlet_Identifier = safe_encode('Outlet_Identifier', Outlet_Identifier)
        enc_Outlet_Size = safe_encode('Outlet_Size', Outlet_Size)
        enc_Outlet_Location_Type = safe_encode('Outlet_Location_Type', Outlet_Location_Type)
        enc_Outlet_Type = safe_encode('Outlet_Type', Outlet_Type)
        enc_Item_Category = safe_encode('Item_Category', Item_Category_class)

        # Derived features
        Price_per_Weight = Item_MRP / Item_Weight if Item_Weight > 0 else 0.0
        Visibility_Ration = Item_Visibility / Item_Weight if Item_Weight > 0 else 0.0
        Outlet_Age = 2025 - Outlet_Establishment_Year

        # Assemble features
        features = np.array([[
            enc_Item_Identifier,
            Item_Weight,
            enc_Item_Fat_Content,
            Item_Visibility,
            enc_Item_Type,
            Item_MRP,
            enc_Outlet_Identifier,
            Outlet_Establishment_Year,
            enc_Outlet_Size,
            enc_Outlet_Location_Type,
            enc_Outlet_Type,
            enc_Item_Category,
            Price_per_Weight,
            Visibility_Ration,
            Outlet_Age
        ]], dtype=float)

        try:
            prediction = model.predict(features)[0]
            st.success("")
            st.markdown(
                f"""
                <div style="background:#0B6E4F;padding:18px;border-radius:12px;color:white">
                  <h2 style="margin:0">Predicted Sales</h2>
                  <h1 style="margin:6px">₹ {prediction:,.2f}</h1>
                  <p style="margin:0.25rem 0 0 0;opacity:0.9">Based on selected item and outlet details.</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with info_col:
    st.markdown("#### Notes")
    st.markdown("- All dropdowns show readable names (no numeric codes).")
    st.markdown("- Unseen items are handled automatically with fallback encoding.")
    st.markdown("- Use the sensitivity chart to explore impact of price & visibility.")

# -------------------------
# Optional sensitivity chart
# -------------------------
if show_chart:
    st.write("---")
    st.subheader("Sales sensitivity: MRP vs Visibility")
    mrp_values = np.linspace(mrp_min, mrp_max, 25)
    vis_values = np.linspace(vis_min, vis_max, 25)
    data_chart = []
    for mrp in mrp_values:
        for vis in vis_values:
            Price_per_Weight_tmp = mrp / Item_Weight if Item_Weight > 0 else 0.0
            Visibility_Ratio_tmp = vis / Item_Weight if Item_Weight > 0 else 0.0
            feat_tmp = np.array([[
                enc_Item_Identifier,
                Item_Weight,
                enc_Item_Fat_Content,
                vis,
                enc_Item_Type,
                mrp,
                enc_Outlet_Identifier,
                Outlet_Establishment_Year,
                enc_Outlet_Size,
                enc_Outlet_Location_Type,
                enc_Outlet_Type,
                enc_Item_Category,
                Price_per_Weight_tmp,
                Visibility_Ratio_tmp,
                Outlet_Age
            ]], dtype=float)
            try:
                pred_tmp = model.predict(feat_tmp)[0]
            except:
                pred_tmp = np.nan
            data_chart.append([mrp, vis, pred_tmp])
    df_chart = pd.DataFrame(data_chart, columns=['MRP', 'Visibility', 'Predicted_Sales'])
    chart = alt.Chart(df_chart).mark_circle(size=90).encode(
        x='MRP:Q',
        y='Visibility:Q',
        color=alt.Color('Predicted_Sales:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['MRP', 'Visibility', alt.Tooltip('Predicted_Sales', format=",.2f")]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# -------------------------
# Footer: input meanings
# -------------------------
st.write("---")
st.markdown("### Input Meaning")
st.markdown("""
- **Item Identifier** — select the specific product.
- **Item Weight** — product weight (kg) for price-per-weight calculation.
- **Item Fat Content** — Low Fat / Regular / Non-Edible.
- **Item Visibility** — fraction (0–1) representing shelf visibility.
- **Item Type** — category like Dairy, Snacks, etc.
- **Item MRP** — retail price.
- **Outlet Identifier** — store ID.
- **Outlet Size / Location / Type** — store attributes affecting sales.
- **Item Category** — Foods / Non-Consumables / Drinks / others.
""")
