"""
This script is used to train the logistic regression classifier on the rice dataset
Note that gradescope will not run this script. However you need to upload this scipt
or a similar script to gradescope for us to evaluate your report and plots
"""

import numpy as np
import matplotlib.pyplot as plt
from logistic import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

if __name__== "__main__":
    # Load the Rice dataset
    data = pd.read_csv('Rice.csv')

    y = data['Class']
    X = data.drop(['Class'], axis=1)

    # Convert y to binary values
    y = y.map({'Cammeo': 1, 'Osmancik': 0}).values

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize weights
    w_init = np.zeros(X_train.shape[1])

    # Train the logistic regression classifier
    w_trained, losses = LogisticRegression.train(X_train, w_init, y_train, epochs=1000, lr=0.1)

    # Predict on test set
    y_test_pred = LogisticRegression.predict(X_test, w_trained)
    y_test_score = LogisticRegression.predict_score(X_test, w_trained)

    # Calculate accuracy
    accuracy = np.mean(y_test_pred == y_test)
    print('Accuracy:', accuracy)

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_score)
    roc_auc = auc(fpr, tpr)

    print('AUC:', roc_auc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
