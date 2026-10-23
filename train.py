"""Train a KNN classifier on scikit-learn's handwritten digits dataset.

Saves the fitted model and scaler to knn_model.pkl and scaler.pkl, which
flaskapp.py loads at startup. Prints test-set accuracy.
"""
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def train(random_state: int = 0):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=random_state
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    accuracy = knn.score(X_test_scaled, y_test)
    return knn, scaler, accuracy


if __name__ == "__main__":
    model, scaler, accuracy = train()
    joblib.dump(model, "knn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print(f"Test accuracy: {accuracy:.4f}")
