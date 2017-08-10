# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:51:51 2017

@author: julio
"""
import numpy as np
import re

#Um exemplo do modelo naive bayes
#Lembrar de mudar os nomes das variaveis e funções quando
#me aprofundar nesse algoritmo.

#Usamos essa função para criar nosso dataset
def loadDataSet():
    postingList = postingList=[['my', 'dog', 'has', 'flea', \
'problems', 'help', 'please'],
['maybe', 'not', 'take', 'him', \
'to', 'dog', 'park', 'stupid'],
['my', 'dalmation', 'is', 'so', 'cute', \
'I', 'love', 'him'],
['stop', 'posting', 'stupid', 'worthless', 'garbage'],
['mr', 'licks', 'ate', 'my', 'steak', 'how',\
'to', 'stop', 'him'],
['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0,1,0,1,0,1] # 1 is abuse 0 not.
    return postingList,classVec


#Essa vunção cria um vetor de vocabulario
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#Essa conta verifica a ocorrencia de palavras no texto
#e retorna um vetor de ocorrencia
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

#Essa função calcula a probabilidade condicional
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  #change to log()
    p0Vect = np.log(p0Num/p0Denom)  #chnage to log()
    return p0Vect,p1Vect,pAbusive
    
    
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1 - pClass1)
    
    if p1 > p0:
        return 1
    else:
        return 0
        
def testingNB():
    listOposts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)
    
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(bagOfWords2VecMN(myVocabList,postinDoc))
    
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
        


emailText = open('C:/Datasets/MachineLearningInAction/Ch04/email/spam/6.txt').read()

regEx = re.compile("\\W*")

listOfTokens = regEx.split(emailText)

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
    
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('C:/Datasets/MachineLearningInAction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('C:/Datasets/MachineLearningInAction/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)

    
        









