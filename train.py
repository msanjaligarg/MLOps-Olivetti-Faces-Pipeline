import numpy as np
import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os


def load_and_split_data():
    print("Loading Olivetti faces dataset...")

    # Load the dataset
    data = fetch_olivetti_faces()
    X = data.images
    y = data.target

    # flatten images for scikit-learn
    X = X.reshape((X.shape[0], -1))  # shape (400, 4096)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Split data: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("\nTraining DecisionTreeClassifier...")

    # Initialize and train the model
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    return model

def save_model(model, filename='savedmodel.pth'):
    print(f"\n Saving model to {filename}...")
    os.makedirs('models', exist_ok=True)
    full_path = os.path.join('models', filename)
    joblib.dump(model, full_path)
    print("Model saved successfully!")


def main():
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    model_path = save_model(model, 'savedmodel.pth')

    # Save test data for later use
    test_data_path = os.path.join('models', 'test_data.pkl')
    joblib.dump((X_test, y_test), test_data_path)
    print(f"Test data saved to {test_data_path}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
