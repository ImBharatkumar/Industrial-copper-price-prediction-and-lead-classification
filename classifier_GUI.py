import streamlit as st
import pandas as pd
import pickle

# Specify the path to the pickle file
pickle_file_path = r'C:\Users\Barry\Desktop\projects\industrial copper modeling\classifier.pkl'  # Replace 'path_to_pickle_file' with the actual path to your pickle file

# Load the model from the pickle file
with open(pickle_file_path, 'rb') as file:
    model = pickle.load(file)

 # Replace 'path_to_model' with the actual path to your trained model

# Define the feature names
feature_names = ['application', 'product_ref', 'selling_price', 'item type_IPL',
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
        inputs[feature] = st.text_input(feature)

    # Create a DataFrame with the input values
    input_df = pd.DataFrame([inputs])

    # Make predictions and display the results
    if st.button("Predict"):
        predictions = predict_class(input_df)
        st.write("Predicted Class:", predictions[0])

# Run the app
if __name__ == '__main__':
    main()

