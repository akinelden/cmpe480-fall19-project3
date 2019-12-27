# Introduction to Artificial Intelligence - Project #3
## Decision Tree Algorithm Implementation

In this project, decision tree algorithm is implemented using Python with [Iris](https://archive.ics.uci.edu/ml/datasets/iris) dataset.

### Project Description

**Dataset:** 
- [UCI Machine Learning Repository Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)

**Aim:**
- Implement decision learning algorithm and compare performances of
information gain and Gini impurity.

**Analysis:**
- Download the dataset
- For 10 times:
   - Shuffle the data
   - Divide your dataset into training (20%), validation (40%) and test
(40%) sets
   - Apply decision tree learning using the training set. Stop splitting
based on the loss in the validation set.
   - Plot the change in training and validation loss given depth of the
trees during training
   - Report the loss in the test set.
- Provide a final comparison between performances of information gain
and Gini impurity metrics and plot/provide the mean and variance in
error for these two different metrics.