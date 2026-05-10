import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Page config ----------
st.set_page_config(page_title="Real Estate Advisor", layout="wide")

st.title("Real Estate Investment Advisor")

# ---------- Load models (cached for speed) ----------
@st.cache_resource
def load_models():
    clf = joblib.load("classifier_model.pkl")
    reg = joblib.load("regression_model.pkl")
    cols = joblib.load("feature_columns.pkl")
    return clf, reg, cols

clf_model, reg_model, feature_columns = load_models()

# ---------- Sidebar Inputs ----------
st.sidebar.header("Input Features")

bhk = st.sidebar.slider("BHK", 1, 5)
size = st.sidebar.number_input("Size in SqFt", min_value=100.0, value=1000.0)
price = st.sidebar.number_input("Current Price (Lakhs)", min_value=1.0, value=50.0)

floor = st.sidebar.number_input("Floor No", min_value=0, value=1)
total_floors = st.sidebar.number_input("Total Floors", min_value=1, value=5)

parking = st.sidebar.selectbox("Parking Space", [0, 1])
schools = st.sidebar.slider("Nearby Schools", 0, 10, 2)
hospitals = st.sidebar.slider("Nearby Hospitals", 0, 10, 2)

# ---------- Prepare input ----------
input_data = {
    'BHK': bhk,
    'Size_in_SqFt': size,
    'Price_in_Lakhs': price,
    'Floor_No': floor,
    'Total_Floors': total_floors,
    'Parking_Space': parking,
    'Nearby_Schools': schools,
    'Nearby_Hospitals': hospitals
}

# Fill missing columns
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

input_df = pd.DataFrame([input_data])
input_df = input_df[feature_columns]
input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# ---------- Input Summary ----------
st.markdown("---")
st.subheader("Input Summary")
st.write(input_df.head(1))

# ---------- Warning system ----------
if price > 300:
    st.warning("High-priced property: Investment risk may be higher")

if size < 500:
    st.warning("Small property size: May limit resale value")

# ---------- Layout ----------
col1, col2 = st.columns(2)

# ---------- Prediction ----------
with col1:
    st.subheader("Prediction")

    if st.button("Predict"):

        try:
            pred_class = clf_model.predict(input_df)[0]
            pred_price = reg_model.predict(input_df)[0]

            # Confidence
            if hasattr(clf_model, "predict_proba"):
                confidence = clf_model.predict_proba(input_df)[0][pred_class]
            else:
                confidence = 0.0

            if pred_class == 1:
                st.success(f"Good Investment (Confidence: {round(confidence*100,2)}%)")
            else:
                st.error(f"Not a Good Investment (Confidence: {round(confidence*100,2)}%)")

            st.info(f"Estimated Price after 5 years: ₹ {round(pred_price, 2)} Lakhs")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ---------- Feature Importance ----------
with col2:
    st.subheader("Top Feature Importance")

    try:
        importances = clf_model.feature_importances_

        feature_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(feature_df['Feature'], feature_df['Importance'])
        ax.invert_yaxis()

        st.pyplot(fig)

    except:
        st.warning("Feature importance not available")

# ---------- Price Trend ----------
st.markdown("---")
st.subheader("Price Growth Trend (5 Years Projection)")

trend = [price * (1.08 ** i) for i in range(6)]
years = ["Now", "1Y", "2Y", "3Y", "4Y", "5Y"]

trend_df = pd.DataFrame({
    "Year": years,
    "Price": trend
})

st.line_chart(trend_df.set_index("Year"))

# ---------- Insights ----------
st.markdown("---")
st.subheader("Insights")

st.write("""
- Price and size are the strongest factors affecting investment decisions.
- Most location-based features showed weak influence due to uniform dataset distribution.
- Model relies heavily on price-related features for prediction.
""")

# ---------- Footer ----------
st.markdown("---")
st.write("This application predicts investment potential and estimates future property price using machine learning.")