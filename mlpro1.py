import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
df= pd.read_csv("auto-mpg.csv", sep=r"\s+", names=columns, usecols= range(8), na_value="?")
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
