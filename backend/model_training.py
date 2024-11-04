import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

data = pd.read_csv("diabetes.csv")


X = data[['Age', 'Glucose', 'Insulin', 'BMI']]
y = data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_pred_nb = naive_bayes_model.predict(X_test)


perceptron_model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
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


with open("naive_bayes_model.pkl", "wb") as nb_file:
    pickle.dump(naive_bayes_model, nb_file)

with open("perceptron_model.pkl", "wb") as perc_file:
    pickle.dump(perceptron_model, perc_file)


cv_scores_nb = cross_val_score(naive_bayes_model, X, y, cv=5)
print(f"Naive Bayes Cross-Validation Scores: {cv_scores_nb}")
print(f"Naive Bayes Average Score: {np.mean(cv_scores_nb)}")

cv_scores_perceptron = cross_val_score(perceptron_model, X, y, cv=5)
print(f"Perceptron Cross-Validation Scores: {cv_scores_perceptron}")
print(f"Perceptron Average Score: {np.mean(cv_scores_perceptron)}")