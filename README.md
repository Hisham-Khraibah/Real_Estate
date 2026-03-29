# real_estate_price_prediction
This app has been built using Streamlit and deployed with Streamlit Community Cloud.

https://khra0005-real-estate.streamlit.app/

This application predicts the price of a house based on property features using supervised machine learning (Regression). The goal is to help estimate property values and support real estate decision-making.

## Features
* User-friendly interface powered by Streamlit.
* Input form to enter property details such as number of beds, baths, square footage, etc.
* Predicts the estimated house price.
* Ensures correct feature alignment using saved feature columns.
* Visualizations including:
  * Correlation heatmap
  * Feature importance chart
* Model and feature structure saved for reuse.

## Dataset
The application uses a **Real Estate Dataset**, which includes the following features:
* year_sold
* property_tax
* insurance
* beds
* baths
* sqft
* year_built
* lot_size
* basement
* popular
* recession
* property_age
* property_type (encoded)
* price (target variable)

## Machine Learning Approach
* **Algorithm Used:** Random Forest Regressor
* **Preprocessing:**
  * Missing value handling
  * Feature engineering (e.g., property_age)
  * Encoding categorical variables (property_type)
  * Feature alignment using saved columns
* **Target Variable:**
  * price (continuous value)
* **Evaluation:**
  * Mean Absolute Error (MAE)

## Project Structure
src/
│
├── data/
│   └── make_dataset.py
│
├── features/
│   └── build_features.py
│
├── models/
│   ├── train_model.py
│   ├── predict_model.py
│   ├── RFmodel.pkl
│   └── feature_columns.pkl
│
├── visualization/
│   └── visualize.py
│
main.py
streamlit.py
requirements.txt

## Technologies Used
* **Streamlit**: For building the web application.
* **Scikit-learn**: For regression modeling (Random Forest) and evaluation.
* **Pandas** and **NumPy**: For data preprocessing.
* **Matplotlib** and **Seaborn**: For data visualization.

## How to Run the Project
### 1. Install dependencies
`pip install -r requirements.txt`

### 2. Run the pipeline
`python main.py`

### 3. Run the Streamlit app
`python -m streamlit run streamlit.py`

## Output
* Trained model saved as:
  `models/RFmodel.pkl`
* Feature columns saved as:
  `models/feature_columns.pkl`
* Visualizations displayed during execution