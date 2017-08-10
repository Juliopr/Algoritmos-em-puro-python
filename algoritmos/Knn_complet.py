# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 21:46:43 2016

@author: julio cesar
"""

#primeiro iremos adquirir os dados.

import csv
import random
import math
import operator
import pandas as pd
    
#Essa função prepara os dados.    
def carregarDados(documento, divisor):
    conjunto_de_teste=[]
    conjunto_de_treino=[]
    with open(documento,'r') as csv_documento:
        conjunto_de_dados = list((csv.reader(csv_documento)))
        for x in range(1,len(conjunto_de_dados)):
             for y in range(len(conjunto_de_dados[0])-1):
             #aqui pode ser meio confuso a principio mais é porque dentro.
             # da lista tem uma string ai acessa tanto indice da lista quanto
             # da string.
                 conjunto_de_dados[x][y] = float(conjunto_de_dados[x][y])

             if random.random() < divisor:
                 conjunto_de_treino.append(conjunto_de_dados[x])         
             else:
	              conjunto_de_teste.append(conjunto_de_dados[x])
    return conjunto_de_treino,conjunto_de_teste
#------------------------------------------------------------------------ 
                      
#aqui é calculado as distancias de um ponto em relação a outro.
#Agora iremos fazer a classificação.
def distanciaEuclidiana(instancia, instancia2):     
     distancia = 0
     for x in range(len(instancia2)-1):
         distancia += pow((float(instancia[x]) - float(instancia2[x])), 2)
     return math.sqrt(distancia)
 
#Aqui eu seleciono o vizinho mais proximo. com base em uma amostra.
def obterVizinhos(conjunto_de_treino, instancia_de_teste, k):
    distancias = []
    for x in range(len(conjunto_de_treino)):
        dist_euclidiana = distanciaEuclidiana(instancia_de_teste,conjunto_de_treino[x])
        distancias.append((conjunto_de_treino[x],dist_euclidiana))
    distancias.sort(key=operator.itemgetter(1))
    vizinhos = []
    for x in range(k):
        vizinhos.append(distancias[x][0])
    return vizinhos

#Aqui se pega o primeiro membro da lista do vizinho mais proximo.
def obterResposta(vizinhos):
    votos = {}
    for x in range(len(vizinhos)):
        resposta = vizinhos[x][-1]
        if resposta in votos:
            votos[resposta] += 1
        else:
            votos[resposta] = 1
    votos_organizados = sorted(votos.items(),key=operator.itemgetter(1),reverse=True)
    return votos_organizados[0][0]
 
    
def obterResposta_continuo(vizinhos):
    soma = 0.0
    if len(vizinhos) != 0:
        for x in range(len(vizinhos)):
            resposta = vizinhos[x][-1]
            soma += resposta
        return soma/len(vizinhos)
    else:
        return "Não existem vizinhos"




#Aqui irei medir a precisão dos acertos.
def obterPrecisao(conjunto_de_teste,predicoes):
    correto = 0
    for x in range(len(conjunto_de_teste)):
        if conjunto_de_teste[x][-1] == predicoes[x]:
            correto += 1
    return (correto/float(len(conjunto_de_teste)))*100.0
    

divisor = 0.67
conjunto_de_treino,conjunto_de_teste = carregarDados('facebook.csv',divisor)  

def main(conjunto_de_treino,conjunto_de_teste,divisor,k):
    #prepara os dados.
   
    print('conjunto de treino:' + repr(len(conjunto_de_treino)))
    print('conjunto de teste' + repr(len(conjunto_de_teste)))
    #gera a predição.
    predicoes = []
    k = k
    for x in range(len(conjunto_de_teste)):
        vizinhos = obterVizinhos(conjunto_de_treino,conjunto_de_teste[x],k)
        resultado = obterResposta_continuo(vizinhos)
        predicoes.append(resultado)
        #print ('> predição =' + repr(resultado)+ '    atual=' + repr(conjunto_de_teste[x][-1]))
    precisao = obterPrecisao(conjunto_de_teste,predicoes)
    print("Precisão:    " + repr(precisao) + '%')
 
#main(conjunto_de_treino,conjunto_de_treino,divisor,k=1)