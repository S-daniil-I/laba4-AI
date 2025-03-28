import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

train_data=pd.read_csv(r'D:\РАФ\2 курс\4сем\AI\laba1\processed_titanic.csv')
X=train_data.drop(['PassengerId','Transported'],axis=1)
Y=train_data['Transported']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

def train_and_evaluate(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(Y_test, pred),
        "Precision": precision_score(Y_test, pred),
        "Recall": recall_score(Y_test, pred),
        "F1 Score": f1_score(Y_test, pred)
    }
    return metrics

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    metrics = train_and_evaluate(model, X_train, Y_train, X_test, Y_test)
    print(f"{name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    print()




