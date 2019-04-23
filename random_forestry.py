from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


def immunotherapy():
    df = pd.read_excel('Immunotherapy.xlsx')

    X = df.drop('Result_of_Treatment', axis=1)
    y = df.Result_of_Treatment

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    print("Accuracy on test data:", model.score(X_test, y_test))

    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(4, 2))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def breastCancer():
    df = pd.read_csv('breast-cancer-wisconsin.data')
    print(df.values)
    print(df.describe())
    for label in df.columns:
        plt.hist(df[label], color="Orange")
        plt.xlabel(label)
        plt.ylabel("Number of Occurrences")
        plt.show()
    X = df.drop(['Sample_code_number', 'Class'], axis=1)
    print(X.values)
    y = df.Class
    print(y.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    print("Accuracy on test data:", model.score(X_test, y_test))

    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(4, 2))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

immunotherapy()
breastCancer()
