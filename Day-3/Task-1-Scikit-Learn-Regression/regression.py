import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample Dataset
df = pd.DataFrame({
    "Area":[600,800,1000,1200,1500,1800,2000,2200],
    "Price":[1500000,2000000,2500000,3000000,3600000,4200000,4700000,5200000]
})

print(df)

X = df[["Area"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.25,random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("\nPredicted Prices")
print(prediction)

print("\nMean Squared Error")
print(mean_squared_error(y_test,prediction))

print("\nR2 Score")
print(r2_score(y_test,prediction))

plt.scatter(X,y,color="blue")
plt.plot(X,model.predict(X),color="red")
plt.title("Linear Regression")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()
