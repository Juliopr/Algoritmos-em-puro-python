# -*- coding: utf-8 -*-
"""
Created on Wed May 03 13:06:31 2017

@author: julio
"""
import random
from math import sqrt
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

bubles = make_blobs(n_samples=10000,centers=5,cluster_std=5.0,n_features=4)
data = bubles[0]



##Essa função é para abrir o documento
#def readfile(filename):
#    lines = [line  for line in file(filename)]
#    
#    #Primeira linha são os titulos
#    colnames = lines[0].strip().split('\t')[1:]
#    rownames = []
#    data = []
#    for line in lines[1:]:
#        p=line.strip().split('\t')
#        #primeira coluna em cada linha é o nome da coluna
#        rownames.append(p[0])
#        #O resto são os dados da linha
#        data.append([float(x) for x in p[1:]])
#    return rownames,colnames,data

#Essa função é chamada pearson correlation
def pearson(v1,v2):
    #Simple somas
    sum1 = sum(v1)
    sum2 = sum(v2)
    
    #A soma dos quadrados
    sum1Sq = sum([pow(v,2) for v in v1])
    sum2Sq = sum([pow(v,2) for v in v2])
    
    #Soma dos produtos
    pSum = sum([v1[i]*v2[i] for i in range(len(v1))])
    
    #Calcula r (person score)
    num = pSum - (sum1*sum2/len(v1))
    den = sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den == 0:
        return 0
    return 1.0 - num/den
  
def euclidian(v1,v2):
    """Essa função recebe duas
       listas e retorna a 
       distancia entre elas"""

    #Armazena o quadrado da distancia
    dist = 0.0
    for x in range(len(v1)):
        dist += pow((v1[x] - v2[x]),2)
    
    #Tira a raiz quadrada da soma
    eucli = sqrt(dist)
    return eucli
       
       
  

def Kcluster(data,distance=euclidian,k=4):
    #Determina o valor maximo e minimo para cada valor
    ranges = [(min([row[i] for row in data]),
               max([row[i] for row in data]))
               for i in range(len(data[0]))]
   
   #Cria K centroides aleatorias
    clusters=[[random.random()*(ranges[i][1] - ranges[i][0])+ranges[i][0]
               for i in range(len(data[0]))] for j in range(k)]
    
    lastmatches = None
    for t in range(100):
        bestmatches = [[] for i in range(k)]
    
        #Verifica qual centroide esta mais perto de cada instancia
        for j in range(len(data)):
            row=data[j]
            bestmatche = 0
            for i in range(k):
                d = distance(clusters[i],row)
                if d < distance(clusters[bestmatche],row):
                    bestmatche = i
            bestmatches[bestmatche].append(j)
        #Se o resultado for o mesmo que da ultima vez esta completo
        if bestmatches == lastmatches:
            break
        lastmatches=bestmatche
    
    #Move o centroide para a zona média do cluster
    #no caso teremos 
        for i in range(k):
            avgs=[0.0]*len(data[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(data[rowid])):
                        avgs[m] += data[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i]=avgs
   
    return bestmatches
            
    


cluster = Kcluster(data,k=5)



c1 = data[[cluster[0]]]
c2 = data[[cluster[1]]]
c3 = data[[cluster[2]]]
c4 = data[[cluster[3]]]
c5 = data[[cluster[4]]]



plt.scatter(c1[:,1],c1[:,0],c='r',alpha=0.7)
plt.scatter(c2[:,1],c2[:,0],c='b',alpha=0.7)
plt.scatter(c3[:,1],c3[:,0],c='g',alpha=0.7)
plt.scatter(c4[:,1],c4[:,0],c='y',alpha=0.7)
plt.scatter(c5[:,1],c5[:,0],c='m',alpha=0.7)

    





      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
