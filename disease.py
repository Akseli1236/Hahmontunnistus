import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

disease_x_train = np.loadtxt('disease_X_train.txt')
disease_y_train = np.loadtxt('disease_y_train.txt')

disease_x_test = np.loadtxt('disease_X_test.txt')
disease_y_test = np.loadtxt('disease_y_test.txt')

# Get the mean of training data
train_prediction = np.mean(disease_y_train)
# Create array with the same length as test data
train_predictions = disease_y_test.size * [train_prediction]

# Compare mean to test data
baseline_mse = mean_squared_error(disease_y_test, train_predictions)
print("Method: Baseline MSE:", baseline_mse)

# We create a model for every type of method
# and with the models with training data.
# Then we predict using x test data and finally
# calculate and print the mse
linear_model = LinearRegression()
linear_model.fit(disease_x_train, disease_y_train)
linear_predictions = linear_model.predict(disease_x_test)
linear_mse = mean_squared_error(disease_y_test, linear_predictions)
print("Method: Linear model MSE:", linear_mse)

decision_tree = DecisionTreeRegressor()
decision_tree.fit(disease_x_train, disease_y_train)
tree_predictions = decision_tree.predict(disease_x_test)
tree_mse = mean_squared_error(disease_y_test, tree_predictions)
print("Method: Decision tree regressor MSE:", tree_mse)

random_forest = RandomForestRegressor()
random_forest.fit(disease_x_train, disease_y_train)
forest_predictions = random_forest.predict(disease_x_test)
forest_mse = mean_squared_error(disease_y_test, forest_predictions)
print("Method: Random forest regressor MSE:", forest_mse)
