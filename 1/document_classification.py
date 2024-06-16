'''
This script contains the skeleton code for the sample assignment-1  document classification.
using the kNN classifier. Here you will be using the kNN classifier implemented in the knn.py file
and get hands dirty with a real world dataset
For the optional part of the assignment you can use the sklearn implementation of the tf-idf vectoriser.
'''
import re
from typing import Tuple
import numpy as np
import pandas as pd
from knn import KNNClassifier


class DocumentPreprocessing:
    '''
    Class to process the text data and convert it to a bag of words model
    '''
    def __init__(self,path):
        self.path = path
        self.data = self.load_data()
        self.domain, self.abstract =  self.extract_labels_and_text(self.data)
        self.class_labels = None
        self.generate_labels()
        self.y_train = self.preprocess_labels(self.domain)
        self.X_train = None
        self.vocabulory = None
    def load_data(self):
        '''    
        Extract the data from the csv file and return the data as a pandas dataframe
        '''
    
        # Load the data from the csv file
        #  Extract the classes present in the domain column
        # Complete your code here
        return pd.read_csv(self.path)

    def extract_labels_and_text(self,data : pd.DataFrame()):
        '''
        Extract the classes in the dataset and the text data
        and save them in the domain and abstract variables respectively
        The outputs are the list of classes and the list of text data
        '''
        # Complete your code here
        domain = data['Domain'].tolist()
        abstract = data['Abstract'].tolist()

        return domain, abstract
    
    def generate_labels(self):
        '''
        Use the domain variable to generate the class labels and 
        save them in the self.class_labels variable
        Example : if self.domain = ['ab', 'cds','aab', 'aab', 'ab', 'cds']
        then self.class_labels = ['ab', 'aab', 'cds']
        '''
        # Complete your code here

        self.class_labels = sorted(list(set(self.domain)))

    def preprocess_labels(self, y_train : list()) -> list():
        '''
        From the text based class labels, convert them to integers
        using the labels generated in the generate_labels function
        Examples : if self.domain = ['ab', 'cds','aab', 'aab', 'ab', 'cds']
        then the out put is  [0, 2, 1, 1, 0, 2]
        '''
        # Complete your code here
        return [self.class_labels.index(label) for label in y_train]

    def remove_special_characters(self,word):
        '''
        This function removes the special characters from the word and returns the cleaned word
        '''
        pattern = r'[^a-zA-Z0-9 ]'  # This pattern only keeps the alphanumeric characters
        # Use the re.sub() function to replace matches with an empty string
        cleaned_word = re.sub(pattern, ' ', word)
        return cleaned_word

    def preprocess(self,text: str ) -> list:
        '''
        Function to preprocess the raw text data
        1. Use the function remove_special_characters to remove the special characters
        2. Remove the words of length 1
        3. Convert to lower case
        return the preprocessed text as a list of words
        '''
        # Complete your code here
        cleaned_text = self.remove_special_characters(text)
        words = cleaned_text.lower().split()
        words = [word for word in words if len(word) > 1]
        if not words:
            print(f"Empty processed words for text: {text}")
        return words
    
    def bag_words(self):
        '''
        Function to convert the text data to a bag of words model.
         
        will break the task into smaller parts below to make it easier to
        understand the process
        '''
      
        
        vocabulory = []
        # Get the unique words in the dataset and sort them in alphabetical order
        # Complete your code here
        all_words = []
        for abstract in self.abstract:
            words = self.preprocess(abstract)
            all_words.extend(words)
        self.vocabulory = sorted(list(set(all_words)))

        # Conver the text to a bag of words model
        # Note: the vector contains the count of the words in the text
        X_train = np.zeros((len(self.abstract), len(self.vocabulory)))

        # Complete your code here
        # Hint: use the preprocess function to preprocess the text data
        for i, abstract in enumerate(self.abstract):
            preprocessed_abstract = self.preprocess(abstract)
            for word in preprocessed_abstract:
                if word in self.vocabulory:
                    idx = self.vocabulory.index(word)
                    if i >= len(self.abstract):
                        print(f"Row Index out of bounds! Abstract index: {i}, Max allowed: {len(self.abstract) - 1}")
                    if idx >= len(self.vocabulory):
                        print(f"Column Index out of bounds! Word index: {idx}, Max allowed: {len(self.vocabulory) - 1}")
                    if i < len(self.abstract) and idx < len(self.vocabulory):
                        X_train[i][idx] += 1

        self.X_train = X_train

    def transform(self,text: list() ) -> np.array:
        '''
        The function takes a list of text data and outputs the 
        feature matrix for the text data.
        Examples if the text is ['this is a test', 'this is another test']
        The output is a numpy array of shape (2, len(self.vocabulory))
        '''
        
        # Complete your code here
        text_matrix = np.zeros((len(text), len(self.vocabulory)))
        for i, text in enumerate(text):
            words = self.preprocess(text)
            for word in words:
                if word in self.vocabulory:
                    text_matrix[i][self.vocabulory.index(word)] += 1

        return text_matrix





if __name__ == '__main__':
    # Make sure to change the path to appropriate location where the data is stored
    trainpath  = './data/webofScience_train.csv'
    testPath = './data/webofScience_test.csv'


    # Create an object of the class document_preprocessing
    document = DocumentPreprocessing(trainpath)
    document.load_data()
    document.bag_words()

    # Some test cases to check if the implementation is correct or not 
    # Note: these case only work for the webofScience dataset provided 
    # You can comment this out this section when you are done with the implementation
    if(document.vocabulory[10] == '0026'):
        print('Test case 1 passed')
    else:
        print('Test case 1 failed')

    if(document.vocabulory[100] == '135'):
        print('Test case 2 passed')
    else:
        print('Test case 2 failed')

    if(document.vocabulory[1000] == 'altitude'):
        print('Test case 3 passed')
    else:
        print('Test case 3 failed')


    # First 10 words in the vocabulory are:
    #['000', '00005', '0001', '0002', '0004', '0005', '0007', '001', '0016', '002']

    print(document.vocabulory[:10])

    pd_Test = pd.read_csv(testPath)

    domain_test, abstract_test = document.extract_labels_and_text(pd_Test)
    y_test = document.preprocess_labels(domain_test)
    X_test = document.transform(abstract_test)

    # Create a kNN classifier object
    knn = KNNClassifier(k=3)

    # Train the kNN classifier
    knn.train(document.X_train, np.array(document.y_train))

    # Compute accuracy on the test set
    accuracy = knn.compute_accuracy(X_test, np.array(y_test))

    # Print the accuracy should be greater than 0.3
    print('Accuracy of the classifier is ', accuracy)

    # For the optional part of the assignment
    # You can use the sklearn implementation of the tf-idf vectoriser
    # The documentation can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # using the following code to tune the hyperparameters
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # Splitting the data into training and validation sets
    X_train_full, y_train_full = document.X_train, np.array(document.y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2)

    accuracies = []

    # Tuning number of neighbors
    k_values = list(range(1, 21))
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.train(X_train, y_train)
        acc = knn.compute_accuracy(X_val, y_val)
        accuracies.append(acc)

    # Plotting accuracy vs number of neighbors
    plt.plot(k_values, accuracies)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy vs. Number of Neighbors')
    plt.show()

    best_k = k_values[accuracies.index(max(accuracies))]
    print(f"Best number of neighbors (k) is: {best_k}")

    # Now, tune subset of training data
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    subset_accuracies = []

    for fraction in fractions:
        subset_size = int(fraction * len(X_train_full))
        X_subset, _, y_subset, _ = train_test_split(X_train_full, y_train_full, train_size=669)
        
        knn = KNNClassifier(k=best_k)
        knn.train(X_subset, y_subset)
        acc = knn.compute_accuracy(X_val, y_val)
        subset_accuracies.append(acc)

    best_fraction = fractions[subset_accuracies.index(max(subset_accuracies))]
    print(f"Best fraction of training data is: {best_fraction * 100}%")
