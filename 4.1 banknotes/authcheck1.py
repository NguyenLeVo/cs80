# Write a program to check if a note is legitimate or not
# Use scikit-learn library to perform supervised machine learning
# The type of problem is classification
# Improve upon authcheck0 with built-in functions in scikit-learn to shorten codes
# Supervised Learning - Classification

import csv
import random

# Import scitkit-learn lib
from sklearn.model_selection import train_test_split
from sklearn import svm # Support Vector Machine, a classification model
from sklearn.linear_model import Perceptron # Perceptron classification model
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes model
from sklearn.neighbors import KNeighborsClassifier # K Nearest Neighbor Classification

'''
Use either one of these four models
model = Perceptron()
model = svm.SVC()
model = KNeighborsClassifier(n_neighbors=3) # 3 Nearest Neighbors
model = GaussianNB()
'''

# Let's use K Nearest Neighbor Model
model = KNeighborsClassifier(n_neighbors=3)

# Read data in from file
with open("banknotes.csv") as file:
    reader = csv.reader(file)
    next(reader)

    # Set up an array with dictionaries for every node in the array
    data= []
    for row in reader:

        '''
        In the banknote.csv, the 5th row represents whether the note is
        real or not. If it is real, then it is 0. Else it is 1.
        '''
        data.append({

            # 1st key: Load all the data from from 1st column to 5th
            "evidence": [float(cell) for cell in row[:4]],

            # 2nd key: Authentic not or Counterfeit note
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

'''
Seperate data into training and testing groups using k-fold cross-validation
and using train_model_split.
The % ratio of data get chosen to train (in this case 75% of the data)
'''
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]

X_training, X_testing, Y_training, Y_testing = train_test_split(
    evidence, labels, test_size=0.25
)

# Fit model using training set
model.fit(X_training, Y_training)

# Predict results using testing set
predictions = model.predict(X_testing)

# Compute how well we performed
correct = (Y_testing == predictions).sum()
incorrect = (Y_testing != predictions).sum()
total = len(predictions)

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Percent accuracy: {100 * correct / total:.2f}%")

'''
RESULTS:

- 1-Nearest Neighbor Classifier (75% training data)
Results for model KNeighborsClassifier
Correct: 343
Incorrect: 0
Percent accuracy: 100.00%

- 3-Nearest Neighbor Classifier (75% training data)
Results for model KNeighborsClassifier
Correct: 343
Incorrect: 0
Percent accuracy: 100.00%

- Perceptron (75% training data)
Results for model Perceptron
Correct: 337
Incorrect: 6
Percent accuracy: 98.25%

- Support Vector Classifier (75% training data)
Results for model SVC
Correct: 341
Incorrect: 2
Percent accuracy: 99.42%

- Gaussian Naive Bayes Classifier (75% training data)
Results for model GaussianNB
Correct: 290
Incorrect: 53
Percent accuracy: 84.55%
'''
