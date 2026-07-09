import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Area":[1000,1200,1500,1800,2000],
    "Price":[150000,180000,220000,260000,300000]
}

df = pd.DataFrame(data)

X = df[["Area"]]
y = df["Price"]

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([[1700]])

print("Predicted House Price:", prediction[0])
