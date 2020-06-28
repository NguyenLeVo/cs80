# Write an AI to predict whether online shopping customers will complete a purchase.
# Supervised Learning - Classification

import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron  # Perceptron classification model
from sklearn import svm  # Support Vector Machine, a classification model

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Open the file
    with open("shopping.csv") as file:
        reader = csv.reader(file)
        
        # Skip the header
        next(reader)
        
        # Load shopping data 
        data = []
        for row in reader:
            
            '''
            You have an abbreviated month name, so use %b:
            from datetime import datetime
            datetime.strptime('Jan', '%b')
            >>> 1
            or
            
            import calendar
            abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
            abbr_to_num['Jan']
            >>> 1
            '''
            # Convert month (11th column) to number:
            
            # In the dataset, every month is abbreviated, except for June. So manually change June
            if row[10] == 'June':
                row[10] = 5
            else:
                abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
                
                # In the month.abbr, the length was 13, element 0 was empty, we want Jan to be 0 not 1 so minus 1
                row[10] = abbr_to_num[row[10]] - 1
            
            # Convert VisitorType (16th column)
            if row[15] == 'Returning_Visitor':
                row[15] = 1
            elif row[15] == 'New_Visitor':
                row[15] = 0
            # 'Other' case
            else:
                row[15] = 0.5
                
            # Convert Weekend (17th column)
            '''
            Shorter
            row[16] = int(row[16] == 'TRUE')
            row[16] = int(row[16] == 'FALSE')
            '''
            if row[16].upper() == 'TRUE':
                row[16] = 1
            elif row[16].upper() == 'FALSE':
                row[16] = 0
            
            # Convert Revenue (18th column)
            '''
            row[17] = int(row[17] == 'TRUE')
            row[17] = int(row[17] == 'FALSE')
            '''
            if row[17].upper() == 'TRUE':
                row[17] = 1
            elif row[17].upper() == 'FALSE':
                row[17] = 0
            
            # Append these values into database
            data.append({
                
                # 1st key: Evidencea: list of lists of values from 1st to 17th data values
                "evidence": [float(cell) for cell in row[:17]],
                
                # 2nd key: Labels: a list of labels of whether there was revenue (18th column)
                # If there's revenue, return 1. Else return 0
                "label": 1 if row[17] == 1 else 0
            })

    # Set up evidence and labels elememts of tuple
    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]
    
    return evidence, labels
    

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    
    return model


# Use y_test to save time. No need to rewrite the evidence and labels tuple to training and testing data set
def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    '''
    This is to predict whether the program's prediction matches the actual result (+ vs + and - vs -)
    correct = (labels == predictions).sum()
    incorrect = (labels != predictions).sum()
    '''
    
    '''
    To check sensivity and specificity, we need to:
    If actual result is true for revenue, we add it to the positive sum counter
    If the predicted result is positive (true for revenue), and it matches the actual result, we add it to the 
    positive counter.
    Vice versa with the negative.
    '''
    pos = 0
    neg = 0
    sum_pos = 0
    sum_neg = 0
    
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            sum_pos += 1
            if actual == predicted:
                pos += 1
        else:
            sum_neg += 1
            if actual == predicted:
                neg += 1

    sensivity = pos / sum_pos

    specificity = neg / sum_neg
    
    return sensivity, specificity
    
    
if __name__ == "__main__":
    main()
    
'''
Perceptron
Correct: 4319
Incorrect: 613
True Positive Rate: 21.21%
True Negative Rate: 99.38%

Support Vector Machine
Correct: 4163
Incorrect: 769
True Positive Rate: 1.54%
True Negative Rate: 100.00%

KNeighbor K=3
Correct: 4217
Incorrect: 715
True Positive Rate: 32.35%
True Negative Rate: 95.47%

KNeighbor K=1
Correct: 4073
Incorrect: 859
True Positive Rate: 39.97%
True Negative Rate: 90.20%
'''