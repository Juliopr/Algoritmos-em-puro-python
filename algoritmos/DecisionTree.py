# -*- coding: utf-8 -*-
"""
Created on Sat May 28 22:16:41 2016

@author: julio cesar
"""

import pandas as pd


#Primeiro crio o documento que sera execultado
my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]
        



#Essa função ira dividir a o conjunto em dois novos conjuntos.
#Iremos utiliza-la para calcular a entropia entre os dois conjuntos.
def divideset(data,column,value):
    #Essa função ira ser nossa função divisora.
   split_function=None
   #Nos iremos usar duas funções lambadas uma para valores númericos e outra para nomes.
   if isinstance(value,int) or isinstance(value,float): #Aqui nos iremos analisar se um valor é float ou int.
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value
   
   # Aqui iremos dividir o conjunto em dois novos conjuntos
   # O lambda não é muito intuitivo a principio mas repare que iremos dividir
   # o conjunto de acordo com a condição dada.
   set1=[row for row in data if split_function(row)]
   set2=[row for row in data if not split_function(row)]
   return (set1,set2)
   
   
#Essa função é bem simples ela ira contar os resultados de cada conjunto.
#Isso ira nos ajudar a calcular a emtropya de cada um.
def uniquecounts(data):
    results = {}
    for row in data:
        #Como o resultado fica na ultima coluna nos não o pegamos.
        r = row[-1]
        if r not in results: 
            results[r] = 0
        results[r]+=1
    return results

#--------------------------------------------------------   
#Etropia ira medir quanto um conjunto é homogenio.
#Pense na entropia como um indice quanto mais alto for
#mais embaralhado sera o conjunto.
def entropy(data):
    from math import log
    #Para quem saca de matematica aqui se
    #aproveitamos da troca de base para achar log2.
    log2 = lambda x:log(x)/log(2)
    results = uniquecounts(data)
    #Aqui é medida a entropia.
    ent = 0.0
    #Aqui iremos pegar cada valor do conjunto.
    for r in results.keys():
        #Repare que calculamos o percentual de ocorrencia do valor.
        #Ou dependendo do contexto a probabilidade de ocorrencia do mesmo.
        p = float(results[r])/len(data)
        #E finalmente acrecentamos ao somatorio.
        ent = ent - p*log2(p)
    return ent
#---------------------------------------------------------

#Essa função é usada para medir a impureza do conjunto.
#O metodo usa probabilidade para isso.
#Atualmente não é tão usado.
def giniImpurity(data):
    total = len(data)
    counts = uniquecounts(data)
    imp=0
    for k1 in counts:
        p1=float(counts[k1])/total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2])/total
            imp += p1*p2
    return imp
#-------------------------------------------------------
#Agora iremos criar uma função para lidar melhor com dados numéricos.
#Esse scoref é ideal para dados numéricos.
def variance(data_set):
    if len(data_set)==0:
        return 0
    data=[float(row[len(row)-1]) for row in data_set]
    mean = sum(data)/len(data)
    variance = sum([(d-mean)**2 for d in data])/len(data)
    return variance
    



#---------------------------------------------------------
#Aqui iremos construir a classe Decisionnode que é onde sera dividido
#os ramos da arvore.
class Decisionnode:
    """Col é o index da coluna que ira ser testada"""
    """value sera o valor que a colunadeverar possuir para ter um
    resultado verdadeiro"""
    """tb e fb são os  decisionnodes, que são os proximos
    nodes na arvore caso os valores sejam verdadeiros ou falsos respectivamente"""
    """results ira armazenar um dicionario com valores do ramo"Branch".
    Aqui não sera armazenado nos de decisão apenas valores finais
    """
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

def buildtree(data,scoref=entropy):

#scoref é o metodo para calcular a heterogeneidade("variedade") do
#conjunto por hora iremos usar a entropia como referencia.
    if len(data) == 0:
        return Decisionnode()

    current_score = scoref(data)
    
    #Essas são as variaveis parametro.
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    
    column_count = len(data[0][:-1]) #conta o número de atributos.
                                   # vamos ate a penultima pois geralmente a ultima é o target.

    for col in range(0,column_count):
        # generate the list of all possible different values in the considered column
        global column_values # Added for debugging
        column_values={}
        for row in data:
            column_values[row[col]]=1
        #Now try dividing the rows up for each value in this column
        for value in column_values.keys():#the 'values' here are the keys of the dictionnary
            (set1,set2) = divideset(data,col,value)
            
            #Information gain
            p = float(len(set1))/len(data) #p is the size of a child set relative to its parents
            gain = current_score-p*scoref(set1)-(1-p)*scoref(set2)#cf.formula information gain
                                                  # é (1-p) por motivos praticos.
            if gain > best_gain and len(set1)>0 and len(set2)>0:#set must not empty
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set1,set2)
    
    #Create the sub branches
    if best_gain > 0:
        trueBranch=buildtree(best_sets[0])
        falseBranch=buildtree(best_sets[1])
        return Decisionnode(col=best_criteria[0],value=best_criteria[1],
                            tb=trueBranch,fb=falseBranch)
    else:
        return Decisionnode(results=uniquecounts(data))
            
                                  
        
    
def classify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
      if v>=tree.value: branch=tree.tb
      else: branch=tree.fb
    else:
      if v==tree.value: branch=tree.tb
      else: branch=tree.fb
    return classify(observation,branch)
                
                                
#Podemos representar a arvore "graficamente".
def printtree(tree,indent=''):
    #isso é uma folha?
    if tree.results != None:
        print (str(tree.results))
    else:
        #Printar o criterio
        print (str(tree.col)+':'+str(tree.value)+'?')
        
        #Printa os ramos
        print(indent + 'T->'),
        printtree(tree.tb, indent+'  '),
        print(indent+'F->'),
        printtree(tree.fb,indent+'  ')
       

#Agora vamos trabalhar em uma função para podar a arvore.
def make_choice(tree):
    import random.randint as rd
    a = tree.results
    a = a.items()
    chose = dict(a[rd.randint(0,len(a))])
    return chose
    
    


def prune(tree,mingain):
    #se os ramos não são folhas podemos podalos.

    if tree.tb != None and tree.tb.results == None:
        prune(tree.tb,mingain)
    if tree.fb != None and tree.fb.results == None:
        prune(tree.fb,mingain)
        
    #Agora vemos se podemos fundir os sub ramos.
    if tree.tb != None and tree.tb.results != None  and tree.fb != None and tree.fb.results != None:
        #Construimos um dataset combinado
        tb,fb = [],[]
        for v,c in tree.tb.results.items():
            tb+=[[v]]*c
        for v,c in tree.fb.results.items():
            fb+=[[v]]*c
    #testa a redução da entropia
        delta = entropy(tb+fb) - (entropy(tb)+entropy(fb))/2
        
        if delta < mingain:
            tree.tb,tree.fb = None,None
            tree.results = uniquecounts(tb+fb)
            
#Vamos criar uma função para classificar conjuntos
#Com algo faltando.
            
def mdclassify(observation,tree):
    if tree.results !=  None:
        return tree.results
    else:
        v=observation[tree.col]
        if v == None:
            tr,fr = mdclassify(observation,tree.tb),mdclassify(observation,tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount)/(tcount + fcount)
            fw = float(fcount)/(tcount + fcount)
            results = {}
            for k,v in tr.items():
                results[k] = v*tw
            for k,v in fr.items():
                results[k] = v*fw
            return results
        else:
            if isinstance(v,(int,float)):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    if v == tree.value:
                        branch = tree.tb
                    else:
                        branch = tree.fb
            return mdclassify(observation,branch)


def make_data(path,traine_prop):
    import random
    data_set = pd.read_csv(path)
    Trainset = []
    Testset = []
    for x in data_set.as_matrix():
        if random.random() < traine_prop:
            Trainset.append(x)
        else:
            Testset.append(x)
    return Trainset,Testset
    




            #API do algoritmo.
#-----------------------------------------------------            
        
train,test = make_data("C:\Datasets\Poker handes\poker-hand-testing.data",0.65)

                         
#tree1 = buildtree(train, scoref=entropy)  


def makeAcuraccy(tree,test):
    erro = 0.0
    for x in range(len(test)):
        data = test[x][:-1]
        a = classify(data,tree)
        b = test[x][-1]
        if a.keys()[0] != b:
            erro += 1
    erro = erro/len(test)
    return 1 - erro


def acuracy_trees(data,tree,iterations):
    plot = []
    prunes = 0.0
    tree2 = tree
    for x in range(iterations):
        true_tree = tree2
        prune(true_tree,prunes)
        plot.append(makeAcuraccy(true_tree,data))
        prunes += 0.01
    return plot

data = pd.read_csv("C:\Datasets\iris.csv")
     
        
        
        
        
        
        
        
        
        
        
        
        
    
    