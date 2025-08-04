import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
import pickle

# Load data
df = pd.read_csv('possum.csv')

# Drop missing values
df = df.dropna(subset=['hdlngth', 'skullw', 'totlngth', 'taill', 'footlgth', 'belly', 'eye', 'age', 'chest'])

# Features and target
X = df[['hdlngth', 'skullw', 'totlngth', 'taill', 'footlgth', 'belly', 'chest', 'age']]
y = df['eye']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('eye_model.sav', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as eye_model.sav")