#%%
import numpy as np
import pandas as pd
from heapq import heappush, heappop
from sklearn import datasets
iris_data = datasets.load_iris()

#%%
feature_reformat = lambda st : st.replace(' (cm)', '').replace(' ', '_')
feature_names = list(map(feature_reformat, iris_data.feature_names))
iris = pd.DataFrame(iris_data.data, columns=feature_names)
iris["target_class"] = iris_data.target

# %%
class Node:
    def __init__(self, impurity, data):
        self.impurity = impurity
        self.data = data
        grouped = data.groupby(["target_class"]).count().iloc[:,0]
        self.dominant_class = grouped.idxmax()
        self.leftChild = None
        self.rightChild = None
        self.order = -1

    def assignOrder(self, order):
        self.order = order

    def split(self, feature, value, impurities):
        self.splitFeature = feature
        self.splitValue = value
        self.leftChild = Node(impurities[0], data[data[feature]<=value])
        self.rightChild = Node(impurities[1], data[data[feature]>value])
        del self.data # for memory efficiency

# %%
def entropy(data):
    n = len(data)
    probs = data.groupby(["target_class"]).count().iloc[:,0] / n
    return - np.sum(probs * np.log2(probs))

def informationGain(parentImpurity, childrenData):
    childEntropies = []
    childSizes = []
    for i in range(len(childrenData)):
        d = childrenData[i]
        childEntropies.append(entropy(d))
        childSizes.append(len(d))
    avgEntropy = np.sum( np.array(childEntropies) * np.array(childSizes) / np.sum(childSizes) )
    return parentImpurity - avgEntropy, childEntropies

# %%
def findBestSplit(data, parentImp, perfMeasure):
    bestFeature = ""
    bestValue = 0
    bestPerf = 0
    bestChildImps = []
    for c in data.columns[:-1]:
        currentData = data[[c, "target_class"]]
        uniques = currentData[c].unique()
        for v in uniques:
            leftChild = currentData[currentData[c] <= v]
            rightChild = currentData[currentData[c] > v]
            performance, childImps = perfMeasure(parentImp, [leftChild, rightChild])
            if performance > bestPerf:
                bestFeature = c
                bestValue = v
                bestPerf = performance
                bestChildImps = childImps
    return bestFeature, bestValue, bestPerf, bestChildImps


def decisionTree(train, validation, performanceMeasure="infGain"):
    impurityMeasure = None
    perfMeasure = None
    if performanceMeasure == "infGain":
        impurityMeasure = entropy
        perfMeasure = informationGain
    elif performanceMeasure == "gini":
        # TODO
        impurityMeasure = None
    else:
        print("Invalid impurity measurement")
        return
    nodeQueue = []
    root = Node(impurityMeasure(train), train)
    heappush(nodeQueue, (root.impurity, root))
    order = 0
    while(True):
        currentNode = heappop(nodeQueue)[1]
        currentNode.assignOrder(order)
        currentData = currentNode.data
        feature, value, perf, childImps = findBestSplit(currentData, currentNode.impurity, perfMeasure)


# %%
