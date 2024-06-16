'''
This script contains the fucntions for assignment2 
'''

import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    '''
    This function contains the evaluation metrics used to evaluate the performance of the 2 class classifier
    '''
    @staticmethod
    def accuracy(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the accuracy of the classifier
        Inputs:
            - y_true : N x 1 ground truth label vector
            - y_pred : N x 1 predicted label vector
        '''
        # Complete your code here
        return np.mean(y_true == y_pred)
    
    @staticmethod    
    def precision(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the precision of the classifier
        Inputs:
            - y_true : N x 1 ground truth label vector
            - y_pred : N x 1 predicted label vector
        '''
        # Complete your code here
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == -1) & (y_pred == 1))
        return tp / (tp + fp)
    
    @staticmethod    
    def recall(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the recall of the classifier
        Inputs:
            - y_true : N x 1 ground truth label vector
            - y_pred : N x 1 predicted label vector
        '''
        # Complete your code here
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == -1))
        return tp / (tp + fn)

    @staticmethod 
    def precision_recall_curve(y_score: np.array, y_true: np.array) -> tuple:
        '''
        Calculates the precision recall curve of the classifier
        Inputs:
            - y_score : N x 1 score vector
            - y_true : N x 1 ground truth label vector
        '''
        # Complete your code here
        precision = []
        recall = []

        thresholds = np.unique(np.sort(y_score))
        for threshold in thresholds:
            y_pred = np.where(y_score >= threshold, 1, -1)
            precision.append(Metrics.precision(y_true, y_pred))
            recall.append(Metrics.recall(y_true, y_pred))
        precision.append(1)
        recall.append(0)
        return (precision, recall, thresholds)
    

    @staticmethod
    def true_positive_rate(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the true positive rate of the classifier
        Inputs:
            - y_true : N x 1 ground truth label vector
            - y_pred : N x 1 predicted label vector
        '''
        # Complete your code here
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == -1))
        return tp / (tp + fn)

    @staticmethod
    def false_positive_rate(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the false positive rate of the classifier
        Inputs:
            - y_true : N x 1 ground truth label vector
            - y_pred : N x 1 predicted label vector
        '''
        
        # Complete your code here
        fp = np.sum((y_true == -1) & (y_pred == 1))
        tn = np.sum((y_true == -1) & (y_pred == -1))
        return fp / (fp + tn)
    
    @staticmethod
    def roc_curve(y_score: np.array, y_true: np.array) -> tuple:
        '''
        Calculates the roc curve of the classifier
        Inputs:
            - y_score : N x 1 score vector
            - y_true : N x 1 ground truth label vector
        '''
        
        # Complete your code here
        fpr = []
        tpr = []

        thresholds = np.unique(y_score)
        thresholds = np.sort(thresholds)[::-1]
        fpr.append(0)
        tpr.append(0)
        for threshold in thresholds:
            y_pred = np.where(y_score >= threshold, 1, -1)
            fpr.append(Metrics.false_positive_rate(y_true, y_pred))
            tpr.append(Metrics.true_positive_rate(y_true, y_pred))

        return (fpr, tpr, thresholds)
    

if __name__ == '__main__':
    # We give simple test cases for each of the functions
    # Note : the library is used only for your use. You are not allowed to use it in your code
    import sklearn.metrics as metrics

    y_true_sample = np.array([1,-1,-1,-1,1,1,1,-1,-1,1])
    y_pred_sample = np.array([1,-1,-1,-1,1,1,-1,1,1,1])
    y_score_sample = np.array([0.9,-0.9,-0.8,-0.8,0.7,0.6,-0.1,0.1,0.2,0.5])


    acc = Metrics.accuracy(y_true_sample, y_pred_sample)
    acc_sklearn = metrics.accuracy_score(y_true_sample, y_pred_sample)


    print('Accuracy is: ', acc)
    print('Accuracy from sklearn is : ', acc_sklearn)

    prec = Metrics.precision(y_true_sample, y_pred_sample)
    prec_sklearn = metrics.precision_score(y_true_sample, y_pred_sample)


    print('Precision is: ', prec)
    print('Precision from sklearn is : ', prec_sklearn)

    rec = Metrics.recall(y_true_sample, y_pred_sample)
    rec_sklearn = metrics.recall_score(y_true_sample, y_pred_sample)

    print('Recall is: ', rec)
    print('Recall from sklearn is : ', rec_sklearn)


    tpr = Metrics.true_positive_rate(y_true_sample, y_pred_sample)
    tpr_sklearn = metrics.recall_score(y_true_sample, y_pred_sample)

    fpr = Metrics.false_positive_rate(y_true_sample, y_pred_sample)

    print('True positive rate is: ', tpr)
    print('True positive rate from sklearn is : ', tpr_sklearn)

    print('False positive rate is: ', fpr)


    # plotting the precision recall curve

    prec, rec, thresholds_custom = Metrics.precision_recall_curve(y_score_sample, y_true_sample)
    prec_sklearn, rec_sklearn, thresholds_sklearn = metrics.precision_recall_curve(y_true_sample, y_score_sample)

    plt.plot(rec, prec,'-*', label='Custom')
    plt.plot(rec_sklearn, prec_sklearn, label='Sklearn')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.show()


    # plotting the roc curve
    fpr, tpr, thresholds_custom = Metrics.roc_curve(y_score_sample, y_true_sample)
    fpr_sklearn, tpr_sklearn, thresholds_sklearn = metrics.roc_curve(y_true_sample, y_score_sample)

    plt.plot(fpr, tpr,'-*', label='Custom')
    plt.plot(fpr_sklearn, tpr_sklearn, label='Sklearn')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()    
    plt.show( )