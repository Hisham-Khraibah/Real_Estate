import pandas as pd
import pickle
import streamlit as st

# Set the page title and description
st.title("Real Estate Price Predictor")
st.write("""
This app predicts the price of a house
based on various property characteristics.
""")

try:
    # Load the pre-trained model
    with open("models/RFmodel.pkl", "rb") as rf_file:
        rf_model = pickle.load(rf_file)

    # Load the feature names used during training
    with open("models/feature_columns.pkl", "rb") as col_file:
        feature_columns = pickle.load(col_file)

    # Prepare the form to collect user inputs
    with st.form("user_inputs"):
        st.subheader("Property Details")

        # Year sold
        year_sold = st.number_input("Year Sold", min_value=1900, max_value=2100, value=2012, step=1)

        # Property tax
        property_tax = st.number_input("Property Tax", min_value=0, value=200, step=1)

        # Insurance
        insurance = st.number_input("Insurance", min_value=0, value=70, step=1)

        # Beds
        beds = st.number_input("Number of Beds", min_value=0, value=3, step=1)

        # Baths
        baths = st.number_input("Number of Baths", min_value=0, value=2, step=1)

        # Square feet
        sqft = st.number_input("Square Feet", min_value=0, value=1500, step=10)

        # Year built
        year_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=2000, step=1)

        # Lot size
        lot_size = st.number_input("Lot Size", min_value=0, value=5000, step=10)

        # Basement
        basement = st.selectbox("Basement", options=[0, 1])

        # Popular
        popular = st.selectbox("Popular", options=[0, 1])

        # Recession
        recession = st.selectbox("Recession", options=[0, 1])

        # Property age
        property_age = st.number_input("Property Age", min_value=0, value=10, step=1)

        # Property type
        property_type_Condo = st.selectbox("Property Type", options=["House", "Condo"])

        # Submit button
        submitted = st.form_submit_button("Predict House Price")

    # Prepare the input and make prediction
    if submitted:
        try:
            property_type_Condo = 1 if property_type_Condo == "Condo" else 0

            # Prepare the input for prediction
            prediction_input = pd.DataFrame([{
                "year_sold": year_sold,
                "property_tax": property_tax,
                "insurance": insurance,
                "beds": beds,
                "baths": baths,
                "sqft": sqft,
                "year_built": year_built,
                "lot_size": lot_size,
                "basement": basement,
                "popular": popular,
                "recession": recession,
                "property_age": property_age,
                "property_type_Condo": property_type_Condo
            }])

            # Reorder the input columns to match the training data
            prediction_input = prediction_input.reindex(columns=feature_columns)

            # Make prediction
            new_prediction = rf_model.predict(prediction_input)

            # Display result
            st.subheader("Prediction Result:")
            st.write(f"Estimated House Price: ${new_prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"Error while making prediction: {e}")

    st.write(
        """We used a machine learning (Random Forest Regression) model to predict the house price.
        The feature importance chart used in this project is shown below."""
    )

    st.image("feature_importance.png")

except Exception as e:
    st.error(f"Error while loading the application: {e}")