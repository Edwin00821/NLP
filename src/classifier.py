import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


class SentimentClassifier:
    """
    Handles the training and evaluation of a Naive Bayes model for sentiment classification.
    """

    def __init__(self):
        self.model = MultinomialNB()

    def train_and_evaluate(self, tfidf_df, labels):
        """
        Trains the Multinomial Naive Bayes classifier and prints evaluation metrics.
        """
        # X features (TF-IDF matrix) and y target (sentiment labels)
        X = tfidf_df.values
        y = labels

        # Split the data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # Train the model
        print("Training Multinomial Naive Bayes model...")
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        print("\n" + "="*40)
        print("🏆 MODEL EVALUATION METRICS")
        print("="*40)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
        print("Detailed Classification Report:")
        print(report)
        print("="*40)

        return accuracy, report
