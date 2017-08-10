# -*- coding: utf-8 -*-
"""
Created on Fri May 05 02:01:11 2017

@author: julio
"""
import random
from math import sqrt
from PIL import ImageDraw,Image


#data = pd.read_csv("C:/Datasets/Apartamentos/apartamentos_JC_0.csv")

#label = list(data.index)
#ata1 = [list(x) for x in data.values]



#Essa função é chamada pearson correlation
#Usamos para calcular determinados tipos de distancia
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
           



#Uma função para visualizar os dados em duas dimensões.
def scaledown(data,distance=pearson,rate=0.01):
    n = len(data)
    
    
    #A distancia real entre todos os pares de instancias
    realdist = [[distance(data[i],data[j]) for j in range(n)]
               for i in range(0,n)]
                   
    #Inicia de maneira aleatoria os pontos em duas dimensões
    loc=[[random.random(),random.random()] for i in range(n)]
    fakedist=[[0.0 for j in range(n)] for i in range(n)]
    
    lasterror=None
    for m in range(0,1000):
        #Encontra as distancias projetadas
        for i in range(n):
            for j in range(n):
                fakedist[i][j] = sqrt(sum([pow(loc[i][x]-loc[j][x],2)
                                      for x in range(len(loc[i]))]))
    
        #Move o  ponto
        grad=[[0.0,0.0] for i in range(n)]
        
        totalerror=0
        for k in range(n):
            for j in range(n):
                if j == k:
                    continue
                #O erro é a diferença percentual entre as distancias
                errorterm = (fakedist[j][k]-realdist[j][k])/realdist[j][k]
                
                #Cada ponto tem que ser movido para mais perto do outro
                #Proporcionalmente com o erro que teve
                grad[k][0] += ((loc[k][0] - loc[j][0])/fakedist[j][k])*errorterm
                grad[k][1] += ((loc[k][1]-loc[j][1])/fakedist[j][k])*errorterm
                
                #Guarda o erro total
                totalerror += abs(errorterm)
        
        
        #Agora avaliamos o erro para saber se é o menor possivel
        if lasterror < totalerror:
            break
        lasterror = totalerror
        for k in range(n):
            loc[k][0] -= rate*grad[k][0]
            loc[k][1] -= rate*grad[k][1]
    return loc
                
#Aqui é importante lembrar que data é 
#a distancia dos pontos já calculada
def draw2d(data,labels,jpeg='mds2d.jpg'):
    img=Image.new('RGB',(2000,2000),(255,255,255))
    draw=ImageDraw.Draw(img)
    for i in range(len(data)):
        x=(data[i][0]+0.5)*1000
        y=(data[i][1]+0.5)*1000
        draw.text((x,y),labels[i],(0,0,0))
    img.save(jpeg,'JPEG')             
           