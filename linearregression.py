import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
df= pd.read_csv("auto-mpg.csv", sep=r"\s+", names=columns, usecols= range(8), na_values="?")
print("First 5 rows of the dataset:")
print(df.head())
print("Missing Values:")
print(df.isnull().sum())
df=df.dropna()
print("Dataset shape after removing missing values")
print(df.shape)
x=df.drop("mpg", axis=1)
y=df["mpg"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Traing Data:", x_train.shape)
print("Testing Data:", x_test.shape)
model=LinearRegression()
model.fit(x_train, y_train)
y_pred= model.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
plt.figure(figsize=(10,6))
plt.plot(y_test.values, marker='o', label='Actual MPG')
plt.plot(y_pred, marker='x', label='Predicted MPG')
plt.xlabel("Test data sample")
plt.ylabel("MPG")
plt.title("Actual vs Predicted MPG")
plt.legend()
plt.grid(True)
plt.show()
