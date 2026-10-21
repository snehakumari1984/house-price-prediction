
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
data = pd.read_csv("house_data.csv")

# 2. Select input features and target
X = data[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
y = data["price_lakhs"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

print("Model training completed!")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2), "lakhs")
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# 6. Save the trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as house_price_model.pkl")
