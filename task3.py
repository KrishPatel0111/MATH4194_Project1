from sklearn . datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC



data = load_breast_cancer()
data_frame = pd.DataFrame(data.data, columns = data.feature_names)
data_frame['label'] = data.target


print(data_frame.head())
print(data_frame.info())
print(data_frame.isnull().sum())
print(data_frame['label'].value_counts())

x = data_frame.drop(columns='label', axis=1)
y = data_frame['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

poly = PolynomialFeatures(degree=2)
x_poly_train = poly.fit_transform(x_train)

pt = PowerTransformer(method='yeo-johnson')
x_power_train = pt.fit_transform(x_train)


kernels = ["linear", "rbf", "poly"]
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    acc = accuracy_score(y_test, y_predicted)
    print(f"SVM with no feature transfromation with {kernel} kernel: Accuracy = {acc:.4f}")
    
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(x_poly_train, y_train)
    x_poly_test = poly.fit_transform(x_test)
    y_predicted = model.predict(x_poly_test)
    acc = accuracy_score(y_test, y_predicted)
    print(f"SVM with polynomial feature transformation with {kernel} kernel: Accuracy = {acc:.4f}")
    
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(x_power_train, y_train)
    x_power_test = pt.fit_transform(x_test)
    y_predicted = model.predict(x_power_test)
    acc = accuracy_score(y_test, y_predicted)
    print(f"SVM with power feature transformation with {kernel} kernel: Accuracy = {acc:.4f}")
    
    


