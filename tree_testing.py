import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Create a hypothetical dataset with categorical features
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red', 'green'],
    'shape': ['circle', 'square', 'circle', 'circle', 'square', 'circle'],
    'label': ['A', 'B', 'A', 'A', 'B', 'A']
})

# Use LabelEncoder to convert categorical features to numerical values
color_encoder = LabelEncoder()
shape_encoder = LabelEncoder()

data['color_encoded'] = color_encoder.fit_transform(data['color'])
data['shape_encoded'] = shape_encoder.fit_transform(data['shape'])

# Split the dataset into features (X) and target (y)
X = data[['color_encoded', 'shape_encoded']]
y = data['label']



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Make predictions on the testing set
y_pred = model.predict(X_test)

# Print the predictions
print("Predictions:", y_pred)


# Assuming the decision tree model has been trained and is named 'model'
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=['color_encoded', 'shape_encoded'], class_names=['A', 'B','C','D'])
plt.show()
