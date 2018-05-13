# coding=utf-8
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 锯齿形状
leafNode = dict(boxstyle="round4", fc="0.8")  # 四个圆角
arrow_args = dict(arrowstyle="<-")  # 箭头形状


def plotNode(nodeTxt, centerPt, parentPt, nodeType):  # 绘图功能,绘制树节点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制带箭头的注释
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')  # 新建绘画窗口
    fig.clf()  # 清空绘图区
    createPlot.ax1 = plt.subplot(111, frameon=False)  # ticks for demo puropses
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)  # 绘制结点(节点文字,箭头终点,箭头起点,节点格式形状)
    plt.show()
