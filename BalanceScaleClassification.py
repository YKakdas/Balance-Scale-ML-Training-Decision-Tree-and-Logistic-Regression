import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def decision_tree_classification():
    # Acquire related fields from the data set
    feature_cols, train_inputs, test_inputs, train_outputs, test_outputs = read_data_set()

    # Create decision tree classifier. Default criteria for quality of splitting, i.e., attribute importance
    # is gini but for sticking with the criterion we've seen in the class, I've used entropy as criterion.
    # For the max_depth, best value depends on the data set. I've tried depths from 2 to 10 for get some insights.
    # Most accuracy I was able to acquire was when depth = 5. That's why I've used max depth as 5.
    decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5)

    # Train decision tree classifier with train data which is 80% of all the dataset.
    decision_tree = decision_tree.fit(train_inputs, train_outputs)

    # Predict the response for test dataset
    predicted_results = decision_tree.predict(test_inputs)

    # Plot resulted decision tree for visualization
    plot_tree(decision_tree=decision_tree, feature_names=feature_cols,
              class_names=["L", "R", "B"], filled=True, precision=4, rounded=True, fontsize=7)

    # Show confusion matrix for visualization of the accuracy
    show_confusion_matrix(test_outputs, predicted_results, "decision tree classification")

    # Print accuracy by comparing test outputs and trained decision tree results
    print("Accuracy for decision tree classification is :", metrics.accuracy_score(test_outputs, predicted_results))


def logistic_regression_classification():
    # Acquire related fields from the data set
    feature_cols, train_inputs, test_inputs, train_outputs, test_outputs = read_data_set()

    # Train logistic regression classifier with train data which is 80% of all the dataset.
    logistic_reg = LogisticRegression()
    logistic_reg = logistic_reg.fit(train_inputs, train_outputs)

    # Predict the response for test dataset
    predicted_results = logistic_reg.predict(test_inputs)

    # Show confusion matrix for visualization of the accuracy
    show_confusion_matrix(test_outputs, predicted_results, "logistic regression classification")

    # Print accuracy by comparing test outputs and trained logistic regression results
    print("Accuracy for logistic regression is :", metrics.accuracy_score(test_outputs, predicted_results))


def read_data_set():
    # https://archive.ics.uci.edu/ml/datasets/Balance+Scale
    # This data set's attributes are categorical, hence it could be used for decision tree and logical regression
    # all column names for the Balance Scale Data Set
    col_names = ['ClassName', 'LeftWeight', 'LeftDistance', 'RightWeight', 'RightDistance']

    # read the data set from csv file and parse according to column names
    data_set = pd.read_csv("balance-scale.csv", header=None, names=col_names)

    # To see how the data set looks like, comment out the next print line
    # print(data_set.info)

    # identify columns as input and output. For given input feature columns
    # find the corresponding classification as ClassName output column

    # input columns of the data set, i.e., feature columns
    # each feature attributes may have values in range 1-5
    feature_cols = ['LeftWeight', 'LeftDistance', 'RightWeight', 'RightDistance']
    inputs = data_set[feature_cols]

    # output column as class name
    # Possible class names are R(Right), L(Left), B(Balanced)
    outputs = data_set.ClassName

    # split data set. Use some portion of inputs and outputs to train the model (train_inputs, test_inputs)
    # Use rest of the data set for evaluating the model (test_inputs, test_outputs)
    # There is no best percentage of dividing the data set but as far as the examples I've seen, it is
    # chosen between 70%-80%. I prefer to split the data set as 80% training and 20% test
    # For the random state, it determines how to divide the data set. Each random_state value gives different data set,
    # however it gives exact same input for each program execution. I want my program to generate same result for each
    # run, that's why I set some integer(0 specifically) instead of None.
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2,
                                                                              random_state=0)
    return feature_cols, train_inputs, test_inputs, train_outputs, test_outputs


def show_confusion_matrix(test_outputs, predicted_results, name):
    # Create confusion matrix to evaluate performance of the model by using test outputs and trained results
    confusion_matrix = metrics.confusion_matrix(test_outputs, predicted_results)

    # Class names as output
    class_names = ["B", "L", "R"]

    # Create a plot
    figure, axes = plt.subplots()

    # Define ticks for axes which are class names
    tick_marks = [0, 1, 2]
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Create heatmap for the confusion matrix(YlGnBu = Yellow -> Green -> Blue colormap)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")

    # Define positions and names of the axes
    axes.xaxis.set_label_position("top")

    axes.set_xticklabels(class_names)
    axes.set_yticklabels(class_names)

    plt.title('Confusion matrix for ' + name)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    # Train dataset via Decision Tree Classification
    decision_tree_classification()

    # Train dataset via Logistic Regression Classification
    logistic_regression_classification()

    # Show visual results for performance of the models
    plt.show()
