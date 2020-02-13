from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


# Read data from an Excel Spreadsheet including data on immunotherapy patients and the results of their treatment and
# use it to classify a model that predicts the success rate for new patients
def immunotherapy():
    df = pd.read_excel('Immunotherapy.xlsx')

    x = df.drop('Result_of_Treatment', axis=1)
    y = df.Result_of_Treatment

    train_test_display(x, y, "immunotherapy")

# Read data from a csv including data on breast cancer patients and the results of their treatment and
# use it to classify a model that predicts the success rate for new patients
def breast_cancer():
    df = pd.read_csv('breast-cancer-wisconsin.data')

    x = df.drop(['Sample_code_number', 'Class'], axis=1)
    y = df.Class

    train_test_display(x, y, "breast cancer")


# Given a data set, builds a random forest classification model to predict subsequent outcomes
# Returns the trained model
def train_test_display(x, y, data_title):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x, y)
    y_predicted = model.predict(x_test)

    print("Accuracy on " + data_title.lower(), " test data:", model.score(x_test, y_test))

    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(6, 3))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted Classification')
    plt.ylabel('Actual Classification')
    plt.title(data_title.title() + ' Classification Accuracy')
    plt.show()

    return model


immunotherapy()
breast_cancer()
