from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.load_dataset import load_data
from sklearn.model_selection import train_test_split
from util import calculate_mae, calculate_mape
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score, mean_absolute_error
from argument import argument_ml

model_name = argument_ml()

# Features와 Labels 선택
features = ['server', 'peer', 'combination', 'send rate', 'server1', 'server2', 'server3', 'server4', 'server5',
            'server6', 'server7', 'server8', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7']
#features = ['server', 'peer', 'combination', 'average_delay','send rate', 'server_1', 'server_2', 'server_3', 'server_4', 'server_5', 'server_6', 'server_7', 'server_8']
labels = ['avg lat', 'TPS']

df = load_data()

X = df[features]
y = df[labels]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_name == 'LR':
    model = LinearRegression()
elif model_name == 'RF':
    model = RandomForestRegressor()
elif model_name == 'SVM':
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
elif model_name == 'KNN':
    model = KNeighborsRegressor(n_neighbors=3)
elif model_name == 'DT':
    model = DecisionTreeRegressor(max_depth=10)


model.fit(X_train, y_train['TPS'])  # Assuming 'TPS' is the column you want to predict
# Make predictions on the test set
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test['TPS'], y_pred)
mse = mean_squared_error(y_test['TPS'], y_pred)
mape = calculate_mape(y_test['TPS'], y_pred)

with open('result.txt', 'a') as f:
    f.write(f'Model name: {model_name}\n')
    # f.write(batch_size)
    f.write(f'Mean Squared Error (TPS): {mse}\n')
    # f.write(f'Mean Squared Error (Latency): {mse_latency}\n')
    f.write(f'Mean Absolute Error (TPS): {mae}\n')
    # f.write(f'Mean Absolute Error (Latency): {mae_latency}\n')
    f.write(f'Mean Absolute Percentage Error (TPS): {mape}\n')
    # f.write(f'Mean Absolute Percentage Error (Latency): {mape_latency}\n')

print(f'Mean Squared Error (TPS): {mse}')
# print(f'Mean Squared Error (Latency): {mse_latency}')
print("MAE (TPS):", mae)
# print("MAE (Latency):", mae_latency)
print("MAPE (TPS):", mape)
# print("MAPE (Latency):", mape_latency)