from digits_pca import get_training_prinicipal_features_and_labels, get_test_prinicipal_features_and_labels
from utils_tree import build_tree, evaluate_tree, plot_contours, plot_roc
from commons import traverse_tree, log
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from config import *
import numpy as np


def evaluate(x_ndimen, decision_tree_root_node, context):
    # evaluate the tree for the training data
    target_actual = [0] * np.alen(x_ndimen)
    target_predicted = [0] * np.alen(x_ndimen)
    target_score = [0] * np.alen(x_ndimen)
    for idx in range(0, np.alen(x_ndimen)):
        target_actual[idx] = x_ndimen[idx][NUM_FEATURES]
        target_predicted[idx], target_score[idx] =\
            evaluate_tree((x_ndimen[idx][:NUM_FEATURES]), decision_tree_root_node)

    # plot the decision boundary
    plot_contours(x_ndimen, target_actual, decision_tree_root_node)
    fpr, tpr, thresholds = roc_curve(target_actual, target_predicted, pos_label=1)
    plot_roc(fpr, tpr, thresholds)

    # get the accuracy for training data
    cm = confusion_matrix(target_actual, target_predicted)
    log("{} Accuracy: {}".format(context, (cm[0][0] + cm[1][1]) / (np.sum(cm))))
    log("{} PPV: {}".format(context, (cm[1][1] / (cm[1][1] + cm[0][1]))))


def main_task():
    # Training
    xi, labels = get_training_prinicipal_features_and_labels()
    labels[labels == NEGATIVE_CLASS] = NEGATIVE_CLASS_MAPPED
    labels[labels == POSITIVE_CLASS] = POSITIVE_CLASS_MAPPED
    x_nd = np.column_stack((xi, labels))

    # build the decision tree
    log("Building decision tree..")
    root_node = build_tree(x_nd)

    # traverse the tree for collecting stats
    stats_dict = {}
    log("Collecting decision tree stats..")
    traverse_tree(root_node, stats_dict)
    log("Decision tree stats (F - Features, T - Targets): ", stats_dict)

    # evaluate the training data
    evaluate(x_nd, root_node, "Training")

    # Testing
    test_xi, test_labels = get_test_prinicipal_features_and_labels()
    test_labels[test_labels == NEGATIVE_CLASS] = NEGATIVE_CLASS_MAPPED
    test_labels[test_labels == POSITIVE_CLASS] = POSITIVE_CLASS_MAPPED
    test_x_nd = np.column_stack((test_xi, test_labels))

    # evaluate the testing data
    evaluate(test_x_nd, root_node, "Testing")

if __name__ == '__main__':
    main_task()
