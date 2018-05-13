from pprint import pprint

import numpy as np


def splitDataset(data_set, featureindex, featurevalue):
    leftSet = data_set[data_set[:featureindex] <= featurevalue]
    rightSet = data_set[data_set[:featureindex] > featurevalue]
    return leftSet, rightSet


def regErr(data_set):
    return np.var(data_set[:, -1])


def chooseBest(data_set):
    m, n = np.shape(data_set)
    Err = regErr(data_set[:, -1])

    bestErr = np.inf
    bestIndex = 0
    bestVal = 0
    for feature_index in range(0, n - 1):
        for feature_value in data_set[:, feature_index]:
            rightSet, leftSet = splitDataset(data_set, feature_index, feature_value)
            tpErr = regErr(rightSet)+regErr(leftSet)
            if tpErr < bestErr:
                bestErr = tpErr
                bestIndex = feature_index
                bestVal = feature_value
        pass


def createTree(train_set):
    chooseBest(train_set)

    return None


if __name__ == '__main__':
    trainSet = np.loadtxt('../data/bikeSpeedVsIq_train.txt')
    testSet = np.loadtxt('../data/bikeSpeedVsIq_test.txt')

    # pprint(np.var(trainSet[:, -1]) * np.shape(trainSet)[0])
    # pprint(np.mean(trainSet[:, -1]))
    # cartTree = createTree(trainSet)
    pprint(
        trainSet[trainSet[:, 0] < 2]
        # trainSet[np.nonzero(trainSet[:, 0] < 2)[0], :]
    )
