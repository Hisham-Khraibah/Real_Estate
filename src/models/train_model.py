from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Function to train the model
def train_model(X, y):

    try:
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the random forest regression model
        model = RandomForestRegressor(
            n_estimators=200,
            criterion='absolute_error',
            random_state=42
        ).fit(X_train, y_train)

        # Create models folder if it does not exist
        os.makedirs('models', exist_ok=True)

        # Save the trained model
        with open('models/RFmodel.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Save the feature names used during training
        with open('models/feature_columns.pkl', 'wb') as f:
            pickle.dump(list(X.columns), f)

        return model, X_test, y_test

    except Exception as e:
        print("Error while training the model:", e)
        return None, None, None