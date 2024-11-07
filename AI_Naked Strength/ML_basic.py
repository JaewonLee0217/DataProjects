import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = {
    '투자액': [1000, 1500, 2000, 2500, 3000],
    '수익' : [100, 150, 200, 250, 300]
}

df = pd.DataFrame(data)

X = df[['투자액']]
y = df['수익']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#training
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')