"""
This assignmet is about implementing features for boundary detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import BSDS300Loader
from sample import Sample
from logistic import LogisticRegression
from feature import FeatureExtractor
from metrics import Metrics
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    from sklearn.metrics import precision_recall_curve

    # Make sure to unzip the dataset in the same folder as this file

    TRAIN_PATH = './BSDS300-images/BSDS300/images/train/'
    TEST_PATH = './BSDS300-images/BSDS300/images/test/'
    SEG_PATH = './BSDS300-human/BSDS300/human/color/'

    N_SAMPLES = 10
    # Number of images to use for training and testing
    # this is used to reduce the computation time
    N_TRAIN = 10
    N_TEST = 2
    FEATURE_DIM = 2  #TODO: change this as it depends on the number of features you use
    N_BINS = 10  # Number of  discretization bins/levels
    BUFFER_SIZE = 10 # Buffer size for the boundary samples 
    # Training parameters
    EPOCHS = 10000
    LR = 0.01

    # Initialize data loaders for training and testing images
    train_loader = BSDS300Loader(TRAIN_PATH,SEG_PATH)
    test_loader = BSDS300Loader(TEST_PATH,SEG_PATH)

    def load_and_preprocess_data(loader, num_images):
        features = []
        labels = []
        feature_extractor = FeatureExtractor(n_bins=N_BINS)
        for img_int in loader.image_list[:num_images]:
            img, img_seg = loader.load_data(img_int)
            img_features = feature_extractor.extract_features(img=img)
            features.append(img_features)
            labels.append(img_seg.flatten())
        return np.array(features), np.array(labels)

    # Load and preprocess training data
    train_features, train_labels = load_and_preprocess_data(train_loader, N_TRAIN)
    train_features_reshaped = train_features.reshape(train_features.shape[0], -1)

    scaler = StandardScaler()
    #train_features_normalized = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1]))
    train_labels = train_labels.reshape(-1)

    # Load and preprocess test data
    test_features, test_labels = load_and_preprocess_data(test_loader, N_TEST)
    test_features_reshaped = test_features.reshape(test_features.shape[0], -1)
    train_features_normalized = scaler.fit_transform(train_features_reshaped)
    test_features_normalized = scaler.transform(test_features_reshaped)
    test_labels = test_labels.reshape(-1)


    classifier = LogisticRegression()
    classifier.train(train_features, train_features_normalized, train_labels, epochs=EPOCHS, lr=LR)

    # Predict on test data
    test_predictions = classifier.predict(test_features_normalized, np.zeros(2))[:, 1]

    # Calculate precision and recall
    precision, recall, _ = Metrics.precision_recall_curve(test_labels, test_predictions)
    accuracy = Metrics.accuracy(test_labels, classifier.predict(test_features_normalized))

    # Plot precision-recall curve
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
