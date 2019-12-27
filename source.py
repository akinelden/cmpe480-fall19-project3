#%%
import numpy as np
import pandas as pd
from sklearn import datasets
iris_data = datasets.load_iris()

#%%
feature_reformat = lambda st : st.replace(' (cm)', '').replace(' ', '_')
feature_names = list(map(feature_reformat, iris_data.feature_names))
iris = pd.DataFrame(iris_data.data, columns=feature_names)
iris["class"] = iris_data.target

# %%
class Node:
    def __init__(self, impurity, data):
        self.impurity = impurity
        self.data = data
        grouped = data.groupby(["class"]).count().iloc[:,0]
        self.decision = grouped.idxmax()
        self.leftChild = None
        self.rightChild = None

    def assignLevel(self, order):
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
    probs = data.groupby(["class"]).count().iloc[:,0] / n
    return - np.sum(probs * np.log2(probs))

def informationGain(parentEntropy, childrenData):
    childEntropies = []
    childSizes = []
    for i in range(len(childrenData)):
        d = childrenData[i]
        childEntropies.append(entropy(d))
        childSizes.append(len(d))
    avgEntropy = np.sum( np.array(childEntropies) * np.array(childSizes) / np.sum(childSizes) )
    return parentEntropy - avgEntropy

# %%