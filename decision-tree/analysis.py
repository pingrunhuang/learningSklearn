# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import subprocess
from sklearn import metrics

import os
print(os.listdir("./input"))

def visualize_tree(tree, feature_names):
    from sklearn.tree import export_graphviz
    with open("dt.dot", "w") as file:
        export_graphviz(tree, out_file=file, feature_names=feature_names)
    command=["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot to produce visualization")


data = pd.read_csv("./input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# drop the unnecessary columns
data=data.drop(columns=["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"])

categorial_columns=["Attrition", "BusinessTravel", "Department", "EducationField", "Gender", 
                   "JobRole", "MaritalStatus", "OverTime"]

for column in categorial_columns:
#     copy_data=data.copy()
    labelEncoder = LabelEncoder()
    data[column]=labelEncoder.fit_transform(data[column])
data.head()


train_data, test_data = train_test_split(data, test_size=0.3)
train_y=train_data["Attrition"]
train_x=train_data.drop("Attrition",axis=1)
test_y=test_data["Attrition"]
test_x=test_data.drop("Attrition",axis=1)


# cls = tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)
cls=tree.DecisionTreeClassifier(criterion="entropy")
cls.fit(X=train_x.as_matrix(), y=train_y)

visualize_tree(cls, [x for x in data.columns if x!="Attrition"])

y_predict = cls.predict(test_x)
print("accuracy:", metrics.accuracy_score(test_y, y_predict))
print("classfication report:", metrics.classification_report(test_y, y_predict))
print("confusion matrix:", metrics.confusion_matrix(test_y, y_predict))
