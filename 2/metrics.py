'''
This script contains the functions for assignment2 
'''

import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    '''
    This function contains the evaluation metrics used to evaluate the performance of a 2 class classifier
    '''
    @staticmethod
    def accuracy(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the accuracy of the classifier
        Inputs:
            - y_true : ground truth label vector of shape N
            - y_pred : predicted label vector of shape N
        '''
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == -1) & (y_pred == -1))
        return (TP + TN) / len(y_true)


    @staticmethod    
    def precision(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the precision of the classifier
        Inputs:
            - y_true : ground truth label vector of shape N
            - y_pred : predicted label vector of shape N
        '''
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == -1) & (y_pred == 1))
        return TP / (TP + FP)


    @staticmethod    
    def recall(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the recall of the classifier
        Inputs:
            - y_true : ground truth label vector of shape N
            - y_pred : 
        '''
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == -1))
        return TP / (TP + FN) 


    @staticmethod 
    def precision_recall_curve(y_score: np.array, y_true: np.array) -> tuple:
        '''
        Calculates the precision recall curve of the classifier
        Inputs:
            - y_score : predicted score vector of shape N
            - y_true : ground truth label vector of shape N

        Note: thresholds are not evaluated as this can be arbitrary
        '''

        precisions = [1.0]
        recalls = [0.0]
        thresholds = [y_score.max() + 1]

        sorted_indices = np.argsort(y_score)[::-1]
        y_score_sorted = y_score[sorted_indices]
        y_true_sorted = y_true[sorted_indices]

        tp = 0
        fp = 0

        for i in range(len(y_score)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp)
            recall = tp / np.sum(y_true)

            precisions.append(precision)
            recalls.append(recall)
            thresholds.append(y_score_sorted[i])

        precisions.append(0.0)
        recalls.append(1.0)
        thresholds.append(y_score.min() - 1)

        return np.array(precisions), np.array(recalls), np.array(thresholds)



    @staticmethod
    def true_positive_rate(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the true positive rate of the classifier
        Inputs:
            - y_true : ground truth label vector of shape N
            - y_pred : predicted label vector of shape N
        '''

        return Metrics.recall(y_true, y_pred)


    @staticmethod
    def false_positive_rate(y_true: np.array, y_pred: np.array) -> float:
        '''
        Calculates the false positive rate of the classifier
        Inputs:
            - y_true : ground truth label vector of shape N
            - y_pred : predicted label vector of shape N

        '''

        FP = np.sum((y_true == -1) & (y_pred == 1))
        TN = np.sum((y_true == -1) & (y_pred == -1))
        return FP / (FP + TN)


    @staticmethod
    def roc_curve(y_score: np.array, y_true: np.array) -> tuple:
        '''
        Calculates the roc curve of the classifier
        Inputs:
            - y_score : predicted score vector of shape N
            - y_true : ground truth label vector of shape N
        Note: thresholds are not evaluated as this can be arbitrary
        '''
        fpr = [0.0]
        tpr = [0.0]
        thresholds = [y_score.max() + 1]

        sorted_indices = np.argsort(y_score)[::-1]
        y_score_sorted = y_score[sorted_indices]
        y_true_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        tp = 0
        fp = 0

        for i in range(len(y_score)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            
            fpr_val = fp / n_neg
            tpr_val = tp / n_pos

            fpr.append(fpr_val)
            tpr.append(tpr_val)
            thresholds.append(y_score_sorted[i])

        fpr.append(1.0)
        tpr.append(1.0)
        thresholds.append(y_score.min() - 1)

        return np.array(fpr), np.array(tpr), np.array(thresholds)


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