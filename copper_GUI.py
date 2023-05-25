import streamlit as st
import pandas as pd
import pickle
import numpy as np


st.title("Industrial Copper Modeling")
st.write("This app performs classification and regression modeling for industrial copper data.")

# Create two columns layout
col1, col2 = st.columns(2)

with col1:
    # Specify the path to the pickle file
    pickle_file_path = r'C:\Users\Barry\Desktop\projects\indus_copper\classifier_model_1.pkl'  # Replace 'path_to_pickle_file' with the actual path to your pickle file

    # Load the model from the pickle file
    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)

     # Replace 'path_to_model' with the actual path to your trained model

    # Define the feature names
    feature_names = ['quantity tons','application','thickness','width', 'product_ref', 'selling_price','delivery_time', 'item type_IPL',
           'item type_PL', 'item type_S', 'item type_SLAWR', 'item type_W',
           'item type_WI']

    # Function to preprocess the input data
    def preprocess_input(input_data):
        # Perform any necessary preprocessing steps
        # E.g., one-hot encoding, scaling, etc.
        processed_data = input_data  # Placeholder for now
        return processed_data

    # Function to make predictions
    def predict_class(input_data):
        processed_data = preprocess_input(input_data)
        predictions = model.predict(processed_data)
        return predictions

    # Streamlit app
    def main():
        # Set app title and description
        st.title("Classification Model")
        st.write("This app predicts the class (1 or 0) based on input features.")

        # Create input fields for feature values
        inputs = {}
        for feature in feature_names:
            inputs[feature] = st.text_input(feature,key=f"name_1_{feature}")

        # Create a DataFrame with the input values
        input_df = pd.DataFrame([inputs])

        # Make predictions and display the results
        if st.button("Predict",key="predict_button_1"):
            predictions = predict_class(input_df)
            st.write("Predicted Class:", predictions[0])

    # Run the app
    if __name__ == '__main__':
        main()






with col2:

    # Specify the path to the pickle file
    pickle_file_path = r'C:\Users\Barry\Desktop\projects\indus_copper\regressor_model_1.pkl'  # Replace 'path_to_pickle_file' with the actual path to your pickle file

    # Load the model from the pickle file
    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)

    # Replace 'path_to_model' with the actual path to your trained model

    # Define the feature names
    feature_names = ['quantity tons', 'application', 'thickness', 'width', 'product_ref', 'item type_IPL',
                     'item type_PL',
                     'item type_S', 'item type_SLAWR', 'item type_W', 'item type_WI']


    # Function to preprocess the input data
    def preprocess_input(input_data):
        # Perform any necessary preprocessing steps
        # E.g., one-hot encoding, scaling, etc.
        processed_data = input_data  # Placeholder for now
        return processed_data


    # Function to make predictions
    def predict_class(input_data):
        processed_data = preprocess_input(input_data)
        predictions = model.predict(processed_data)
        return np.exp(predictions)


    # Streamlit app
    def main():
        # Set app title and description
        st.title("Regression Model")
        st.write("This app predicts copper price based on input features.")

        # Create input fields for feature values
        inputs = {}
        for feature in feature_names:
            inputs[feature] = st.text_input(feature,key=f"name_2_{feature}")

        # Create a DataFrame with the input values
        input_df = pd.DataFrame([inputs])

        # Make predictions and display the results
        if st.button("Predict",key="predict_button_2"):
            predictions = predict_class(input_df)
            st.write("Predicted Class:", predictions[0])


    # Run the app
    if __name__ == '__main__':
        main()

