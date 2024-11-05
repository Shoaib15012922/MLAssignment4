import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle


data = pd.read_csv("diabetes.csv")


X = data[['Age', 'Glucose', 'Insulin', 'BMI']]
y = data['Outcome']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_pred_nb = naive_bayes_model.predict(X_test)



class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        y_ = np.where(y == 0, -1, 1)  # Map 0 to -1 for binary classification

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                # Update rule if prediction is incorrect
                if y_predicted != y_[idx]:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)


    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "n_iter": self.n_iter}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self



perceptron_model = Perceptron(learning_rate=0.01, n_iter=1000)
perceptron_model.fit(X_train, y_train)
y_pred_perceptron = perceptron_model.predict(X_test)



def evaluate_model(y_test, y_pred, model_name):
    print(f"--- {model_name} Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}\n")



evaluate_model(y_test, y_pred_nb, "Naive Bayes")
evaluate_model(y_test, y_pred_perceptron, "Perceptron")


import cloudpickle as cp

with open("naive_bayes_model.pkl", "wb") as nb_file:
    pickle.dump(naive_bayes_model, nb_file)

with open("perceptron_model.pkl", "wb") as perc_file:
    cp.dump(perceptron_model, perc_file)


cv_scores_nb = cross_val_score(naive_bayes_model, X_scaled, y, cv=5)
print(f"Naive Bayes Cross-Validation Scores: {cv_scores_nb}")
print(f"Naive Bayes Average Score: {np.mean(cv_scores_nb)}")


cv_scores_perceptron = cross_val_score(perceptron_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"Perceptron Cross-Validation Scores: {cv_scores_perceptron}")
print(f"Perceptron Average Score: {np.mean(cv_scores_perceptron)}")