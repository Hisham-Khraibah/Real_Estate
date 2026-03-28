# real_estate_price_prediction
This app has been built using Streamlit and can be deployed using Streamlit Community Cloud.

URL:

This application predicts the price of a house based on various property-related features. The model leverages machine learning techniques to estimate real estate prices and help users understand property valuation.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter property details such as size, number of rooms, and other features.
- Real-time prediction of house price based on the trained model.
- Feature importance visualization to understand key factors influencing price.
- Data visualization including correlation heatmaps.

## Dataset
The application uses a dataset containing features such as:
- Year Sold
- Property Tax
- Insurance
- Number of Bedrooms and Bathrooms
- Square Footage (sqft)
- Year Built
- Lot Size
- Basement (Yes/No)
- Property Popularity
- Economic Conditions (Recession indicator)
- Property Age
- Property Type (Condo or not)

## Technologies Used
- **Streamlit**: For building the interactive web application.
- **Scikit-learn**: For model training and evaluation (Random Forest Regression).
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For data visualization and feature analysis.
- **Pickle**: For saving and loading the trained model.