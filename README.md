# Penguins: A Machine Learning Example with the Palmer Penguins Dataset

This repository showcases essential machine learning workflows in Python using the famous Palmer Penguins dataset. The focus is on data preprocessing, exploration, visualization, and classification of penguin species with a `KNeighborsClassifier`.

## Overview

The aim is to demonstrate the basic steps of a data science project—from data exploration and cleaning to model training and evaluation, entirely in Python.

## Requirements

- Python 3.x
- numpy
- pandas
- seaborn
- scikit-learn

To install required packages:

pip install numpy pandas seaborn scikit-learn

## Dataset

The Palmer Penguins dataset provides physiological measurements and contextual information about three Antarctic penguin species: Adelie, Chinstrap, and Gentoo. The dataset can be loaded directly via Seaborn.

<img width="664" height="396" alt="penguinDataset" src="https://github.com/user-attachments/assets/78c392bb-f8e9-446f-95ff-29391240a953" />

## Loading and Inspecting Data

`import numpy as np`

`import pandas as pd`

`import seaborn as sns`

`df = sns.load_dataset("penguins")`

## Exploratory Data Analysis and Visualization

- Overview and handling of missing values (via `dropna`)
- Visualization: pairplot colored by `species` using Seaborn

`sns.pairplot(data=df, hue="species")`

<img width="1098" height="973" alt="penguinPairPlot" src="https://github.com/user-attachments/assets/10636820-5c21-4393-8040-f0e9bbae739b" />


## Feature Engineering

- Remove non-numerical columns (`island`, `sex`)
- Use only numerical features

`df = df.drop(['island', 'sex'], axis=1)`

`df = df.dropna()`

## Modeling

- Target variable: `species`
- Features: bill length/depth, flipper length, body mass

`from sklearn.model_selection import train_test_split`

`X = df.drop("species", axis=1)`

`y = df["species"]`

<img width="501" height="411" alt="features" src="https://github.com/user-attachments/assets/702c1c73-72de-4f9c-bd64-d707a3963b78" />

<img width="320" height="213" alt="labels" src="https://github.com/user-attachments/assets/466167f9-d151-4a1d-8bd3-20488a3ad7a8" />

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)`

`from sklearn.neighbors import KNeighborsClassifier`

`clf = KNeighborsClassifier(n_neighbors=3)`

`clf.fit(X_train, y_train)`

## Evaluation

- Predictions and accuracy

`from sklearn.metrics import accuracy_score`

`predictions = clf.predict(X_test)`

`print(accuracy_score(y_test, predictions)) # about 0.88`

## Results

The KNN model classifies penguin species in the test set with approximately 88% accuracy.

## Contributors

Created by Klaus Fischer.

## References

- Palmer Penguins Dataset: https://allisonhorst.github.io/palmerpenguins/
- Seaborn Documentation: https://seaborn.pydata.org/
- scikit-learn Documentation: https://scikit-learn.org/
