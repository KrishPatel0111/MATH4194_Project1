import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

x = np.linspace(0,10,100).reshape(-1, 1)
#print(x)
e = np.random.normal(0,1,100)
#print(e)
y = 2*x.flatten()*x.flatten() + 3*x.flatten() + e

x_train,x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=2)

model_linear = LinearRegression()
model_linear.fit(x_train,y_train)
y_predicted_linear = model_linear.predict(x_test)

poly = PolynomialFeatures(degree=2)
x_poly_train = poly.fit_transform(x_train)
model_poly  = LinearRegression()
model_poly.fit(x_poly_train, y_train)
x_test_poly = poly.transform(x_test)
y_predicted_poly = model_poly.predict(x_test_poly)

alphas = np.logspace(-4, 4, 100)

ridge_scores = [cross_val_score(Ridge(alpha=a, max_iter=10000), x_poly_train, y_train, cv=5, scoring="neg_mean_squared_error").mean() for a in alphas]
best_alpha_ridge = alphas[np.argmax(ridge_scores)]
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(x_poly_train, y_train)
y_predicted_ridge = ridge_model.predict(x_test_poly)

lasso_scores = [cross_val_score(Lasso(alpha=a, max_iter=10000), x_poly_train, y_train, cv=5, scoring="neg_mean_squared_error").mean() for a in alphas]
best_alpha_lasso = alphas[np.argmax(lasso_scores)]  # Select best alpha
lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_model.fit(x_poly_train, y_train)
y_predicted_lasso = lasso_model.predict(x_test_poly)


mse_linear = mean_squared_error(y_test, y_predicted_linear)
mse_poly = mean_squared_error(y_test, y_predicted_poly)
mse_ridge = mean_squared_error(y_test, y_predicted_ridge)
mse_lasso = mean_squared_error(y_test, y_predicted_lasso)


plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='black', label="Actual Data")
plt.scatter(x_test, y_predicted_linear, color='red', label="Predictions")
plt.title("Linear Regression\nMSE = {:.2f}".format(mse_linear))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='black', label="Actual Data")
plt.scatter(x_test, y_predicted_poly, color='blue', label="Predictions")
plt.title("Polynomial Regression (Degree 2)\nMSE = {:.2f}".format(mse_poly))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='black', label="Actual Data")
plt.scatter(x_test, y_predicted_ridge, color='green', label="Predictions")
plt.title("Ridge Regression\nBest α = {:.4f}, MSE = {:.2f}".format(best_alpha_ridge, mse_ridge))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='black', label="Actual Data")
plt.scatter(x_test, y_predicted_lasso, color='purple', label="Predictions")
plt.title("Lasso Regression\nBest α = {:.4f}, MSE = {:.2f}".format(best_alpha_lasso, mse_lasso))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()