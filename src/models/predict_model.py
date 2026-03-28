# Import regression metrics
from sklearn.metrics import mean_absolute_error

# Function to predict and evaluate
def evaluate_model(model, X_test, y_test):

    try:
        # Predict the house prices on the testing set
        y_pred = model.predict(X_test)

        # Calculate the mean absolute error
        mae = mean_absolute_error(y_test, y_pred)

        return mae

    except Exception as e:
        print("Error while evaluating the model:", e)