import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Data
df = pd.read_csv('data.csv')
print("Data Loaded:")
print(df.head())

X = df[['Area_Sqft','BHK','Location']]
y = df['Price_Lakhs']

# 2. Preprocess + Model Pipeline
preprocessor = ColumnTransformer(
    transformers=[('loc', OneHotEncoder(), ['Location'])],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LinearRegression())
])

# 3. Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test) * 100
print(f"\nModel Trained! Accuracy: {accuracy:.2f}%")

# 4. Prediction Loop
while True:
    print("\n--- Predict House Price (type q to exit) ---")
    try:
        area_input = input("Area_Sqft: ")
        if area_input.lower() == 'q': break
        area = int(area_input)
        bhk = int(input("BHK: "))
        loc = input("Location: ")

        input_data = pd.DataFrame([[area, bhk, loc]], columns=['Area_Sqft','BHK','Location'])
        price_lakhs = pipeline.predict(input_data)[0]

        # Show in Lakhs + Rupees + Crore
        if price_lakhs >= 100:
            print(f"--> Predicted Price: {price_lakhs/100:.2f} Crore ({price_lakhs:.2f} Lakhs)")
        else:
            print(f"--> Predicted Price: {price_lakhs:.2f} Lakhs = Rs {price_lakhs*100000:.0f}")

    except Exception as e:
        print(f"Error: {e}")