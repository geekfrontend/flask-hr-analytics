import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('./models/logistic_regression_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

# Streamlit app interface
def main():
    st.title("Logistic Regression Model Prediction")

    st.write("Enter the features below to make a prediction:")

    # User inputs for features
    try:
        feature_1 = st.number_input("Feature 1", min_value=0.0, step=0.01)
        feature_2 = st.number_input("Feature 2", min_value=0.0, step=0.01)
        feature_3 = st.number_input("Feature 3", min_value=0.0, step=0.01)
        feature_4 = st.number_input("Feature 4", min_value=0.0, step=0.01)

        # Creating a list of features from the user input
        features = np.array([feature_1, feature_2, feature_3, feature_4]).reshape(1, -1)
    except ValueError:
        st.error("Please enter valid numerical values for all features.")
        return

    # Button to trigger prediction
    if st.button("Predict"):
        try:
            # Scale the features
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[:, 1]

            # Show the result
            st.subheader(f"Prediction: {int(prediction[0])}")
            st.subheader(f"Probability: {float(probability[0]):.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
