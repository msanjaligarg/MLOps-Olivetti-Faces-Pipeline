import joblib
from sklearn.metrics import accuracy_score
import numpy as np

def load_model(model_path='savedmodel.pth'):
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    return model

def load_test_data(data_path='test_data.pkl'):
    print(f"Loading test data from {data_path}...")
    X_test, y_test = joblib.load(data_path)
    print(f"Test data loaded: {X_test.shape[0]} samples")
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("="*50)
    
    return test_accuracy

def main():
    try:
        # Load model
        model = load_model()
        
        # Load test data
        X_test, y_test = load_test_data()
        
        # Evaluate model
        test_accuracy = evaluate_model(model, X_test, y_test)
        
        print("\nTesting completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("Please run train.py first to generate the model and test data.")
    except Exception as e:
        print(f"\n Unexpected error: {e}")

if __name__ == "__main__":
    main()
