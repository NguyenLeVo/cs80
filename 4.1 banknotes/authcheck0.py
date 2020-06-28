# Write a program to check if a note is legitimate or not
# Use scikit-learn library to use machine learning functions to model
# Supervised Learning - Classification

import csv
import random

# Import scitkit-learn lib
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
K-fold cross-validation:
Let's pick k = 10. We divide the dataset into 10 equal parts.
We train 9 parts then use the remaining part to test the result.
We repeat the process 10 times, each time use a different part for testing.
Then we average the results from the 10 tests to see how well the model performs.
'''

# Seperate data into training and testing groups using k-fold cross-validation

'''
In this model, we divide the training sets with a training:testing ratio of 75%.
Or in terms of k-fold, we divide it into 4 parts, train 3, and test 1.
Randomize the data to avoid bias
The % ratio of data get chosen to train (in this case 75% of the data)
'''
holdout = int(0.75 * len(data))
random.shuffle(data)
training = data[:holdout]
testing = data[holdout:]

# Train model using training set

'''
Set up arrays for evidence and label

[row["evidence"] for row in training] is the same as:

X_training = []
for row in training:
    X_training = row[evidence]

model.fit:
Using the k nearest neighbor method, we feed our training data to make the model
'''
X_training = [row["evidence"] for row in training]
Y_training = [row["label"] for row in training]

model.fit(X_training, Y_training)

# Validate results using testing set
X_testing = [row["evidence"] for row in testing]
Y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)

# Compute how well we performed
correct = 0
incorrect = 0
total = 0

'''
Y_testing is the actual results from the testing data set
predictions is the predicted results using the method given the training data set
'''
for actual, predicted in zip(Y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

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
