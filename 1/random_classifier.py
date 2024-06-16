'''
This contains the skeleton code for the random classifier
exercise.
'''

import numpy as np
import matplotlib.pyplot as plt
from perceptron import MutiClassPerceptron


def generate_data(N, D, p):
    '''Generate random data and labels'''
    X = np.random.randn(N, D)
    y = np.random.binomial(1, p, N)
    return X, y

if __name__ == '__main__':

    # Generate random data

    N = 100
    p = 0.5
    d_init = 10

    model = MutiClassPerceptron()
    train_accuracies = []
    test_accuracies = []
    # Let #X_train = N x d abd #y_train = N x 1 be the training data
    # Using random.randn() generate X_train and random.binomial() generate y_train
    # make sure to one-hot-encode y_train
    # similarly generate X_test and y_test
    # make sure to one-hot-encode y_test
    # Initialize the perceptron with W_int = np.zeros((d,2)) and train it
    # Repeat the above steps for d = np.arange(10,200,10)
    # Plot the accuracy on the training and test set as a function of d
    for d in range(10, 210, 10):
        X_train, y_train_raw = generate_data(N, d, p)
        X_test, y_test_raw = generate_data(N, d, p)

        y_train = model.one_hot_encode(y_train_raw)
        y_test = model.one_hot_encode(y_test_raw)

        W_init = np.zeros((2, d))
        W, _ = model.train(X_train, y_train, W_init)

        y_train_pred = model.predict(W, X_train)
        y_test_pred = model.predict(W, X_test)

        train_acc = np.mean(np.argmax(y_train_pred, axis=1) == y_train_raw)
        test_acc = np.mean(np.argmax(y_test_pred, axis=1) == y_test_raw)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    plt.plot(range(10, 210, 10), train_accuracies, label="Train Accuracy")
    plt.plot(range(10, 210, 10), test_accuracies, label="Test Accuracy")
    plt.xlabel('Input Dimension (D)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Input Dimension')
    plt.show()







