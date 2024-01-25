import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from joblib import dump, load
from scipy.special import boxcox1p


st.title("Industrial Copper Price prediction")

# Create two columns layout
col1, col2 = st.columns(2)

with col1:
    # Specify the path to the pickle file
    pickle_file_path = os.path.join("artifacts", "rf_model.joblib")  # Replace 'path_to_pickle_file' with the actual path to your pickle file

    # Load the model from the pickle file
    #with open(pickle_file_path, 'rb') as file:
    model = load(pickle_file_path)

     # Replace 'path_to_model' with the actual path to your trained model

    # Define the feature names
    feature_names = ['quantity tons', 'country', 'application', 'thickness', 'width',
       'product_ref', 'delivery_dats']

    # Function to preprocess the input data
    def preprocess_input(input_data, lambdas):
    # Perform any necessary preprocessing steps
        features = ['quantity tons', 'thickness', 'width']
        processed_data = input_data.copy()  # Create a copy of the input data to avoid modifying the original data
        for feature,lam in zip(features,lambdas):
            if feature in input_data:
                original_data = input_data[feature]
                # transform data with previously fitted lambda
                fitted_data = boxcox1p(original_data,lam)
                # Replace the original feature data with the transformed data
                processed_data[feature] = fitted_data
        return list(processed_data.values())



    # Function to make predictions
    def predict_class(input_data):
        lam = [0.16566176182102382,-0.15661990549881208,0.8944320487982019]
        processed_data = preprocess_input(input_data,lam)
        x=pd.DataFrame([processed_data])
        st.write(x)
        predictions = model.predict(x)
        return predictions

    # Streamlit app
    def main():
        # Set app title and description
        st.title("Prediction Model")

        # Create input fields for feature values
        inputs = {}
        for feature in feature_names:
            inputs[feature] = st.number_input(feature,key=f"name_1_{feature}")
        st.write(inputs)
        # Create a DataFrame with the input values
        #input_df = pd.DataFrame([inputs])

        # Make predictions and display the results
        if st.button("Predict",key="predict_button_1"):
            predictions = predict_class(inputs)
            st.write("Predicted price:", np.exp(predictions[0]))

    # Run the app
    if __name__ == '__main__':
        main()