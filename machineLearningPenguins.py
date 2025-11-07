# machineLearningPenguins.py
# A simple machine learning script to classify penguin species using the K-Nearest Neighbors algorithm.
# The script loads the penguins dataset, preprocesses the data, trains the model, and evaluates its accuracy.
# Required libraries: numpy, pandas, seaborn, scikit-learn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the penguins dataset
df = sns.load_dataset("penguins")

# Visualize the pairplot of the dataset
sns.pairplot(data = df, hue = "species")

# Data preprocessing
df.isnull().sum()
# Drop rows with missing values
df = df.dropna()
# Drop non-numeric columns
df = df.drop(["island", "sex"], axis = 1)

# Define features and target variable
X = df.drop(["species"], axis = 1)
# Target variable
y = df["species"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 119)

# Initialize and train the K-Nearest Neighbors classifier
clf = KNeighborsClassifier(n_neighbors = 3)
# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy_score(predictions, y_test)
# Output the accuracy
print("Accuracy:", accuracy_score(predictions, y_test))