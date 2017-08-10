# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:37:36 2017

@author: julio
"""
import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir

#Essa é uma simples função para gerar um conjunto de dados.
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#Aqui irei fazer o KNN essa verção usa numpy pesadamente.
def classify0(inX,dataSet,labels,k):
    """inX: tem que ser uma array de uma dimensão ele é a
       instancia que será classificada.
       dataSet: se trata do conjunto de treino.
       labels:aqui ficam as etiquetas o conjunto de treino,
       é importante lembrar que devem ter o mesmo número de linhas
       que o dataset.
       K: O número de vizinhos se você não sabe o que isso 
       siguinifica é porque você deve dar uma estudada antes
       de usar a porra desse algoritmo o ze ruela.""" 
       
    #Nas linhas abaicho calculamos a distancia.
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    #Nas linhas abaicho selecionamos os k mais proximos.
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    

#Essa função é responsavel por abrir o file.
#Ela o adapta de modo que classify0() possa ler.
#Nivel:Porca.
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
#Agora criaremos uma função para normalizar os dados
def autoNorma(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals    

#Criaremos aqui uma função para medir o desempenho do algoritmo
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("C:/Datasets/MachineLearningInAction/ch02/datingTestSet2.txt")
    normMat,ranges,minVals = autoNorma(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifyResults = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                    datingLabels[numTestVecs:m],3)
        print "The classifier came bach with: %d, the real answer is: %d"\
                % (classifyResults,datingLabels[i])
        if (classifyResults != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
 
#Uma função especifica para classificar a galera.   
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentagem \
    of time playng video games?"))
    ffMiles = float(raw_input("frequent flier miles\
    earned per years?"))
    icecream = float(raw_input("liter of ice cream\
    consumed per years?"))
    datingDataMat,datingClassLabels = file2matrix("C:/Datasets/MachineLearningInAction/ch02/datingTestSet2.txt")
    norMat,ranges,minVals = autoNorma(datingDataMat)
    inArr = np.array([ffMiles,percentTats,icecream])
    classifierResults = classify0((inArr-minVals)/ranges,
                                  norMat,datingClassLabels,3)
    print "You will probably like this person:",\
        resultList[classifierResults - 1]

#Essa função transforma as array 32x32 para 1x1024
def img2vector(filename):
    returnVector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector
    
#Agora vamos criar uma função para classificar os digitos
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('C:/Datasets/MachineLearningInAction/Ch02/trainingDigits/')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('C:/Datasets/MachineLearningInAction/Ch02/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('C:/Datasets/MachineLearningInAction/Ch02/testDigits/')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('C:/Datasets/MachineLearningInAction/Ch02/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with: %d, the real valor is: %d"\
                % (classifierResult,classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    

            
 

def make_plot(x,y,labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x[:,0], y[:,1],
           15.0*np.array(labels), 
           15.0*np.array(labels))

   
    