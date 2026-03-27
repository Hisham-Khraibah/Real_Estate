from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/final.csv"
    df = load_and_preprocess_data(data_path)

    # Plot the correlation heatmap
    plot_correlation_heatmap(df)

    # Create features and separate features and target
    X, y = build_features(df)

    # Train the random forest regression model
    model, X_test, y_test = train_model(X, y)

    # Evaluate the model
    plot_feature_importance(model, X)
    mae = evaluate_model(model, X_test, y_test)
    print(f"Mean Absolute Error: {mae}")