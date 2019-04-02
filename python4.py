import numpy as np
import pandas as pd
import time
import csv
from scipy.spatial import distance
import networkx as nx
import matplotlib.pyplot as plt
from time import time 
import statistics
import random
from datetime import datetime
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison





aux=0
matriz=np.arange(1800*5,dtype=float).reshape(1800,5)
ordenes=[27,81,243,729]
for i in range((len(ordenes))*4):
    
    if i <=3:
        rango=random.randint(ordenes[i]*3,ordenes[i]*ordenes[i])
        G=nx.dense_gnm_random_graph(ordenes[i],rango)
        graf=1
    if 3<i<=7:
       rango=random.randint(1,9) 
       G=nx.gnp_random_graph(ordenes[i-4],(rango/10))
       graf=2
    if 7<i<=11:
        rango=random.randint(ordenes[i-8]*2,ordenes[i-8]*ordenes[i-8])
        G=nx.gnm_random_graph(ordenes[i-8],rango , seed=None, directed=False)   
        graf=3
    for j in range(10):
        lista=[]
        lista[:]=G.edges
        
        for r in range(len(lista)):
            R=np.random.normal(loc=20, scale=5.0, size=None)
            G.add_edge(lista[r][0],lista[r][1],capacity=R)
            
        for k in range(15):
            tiempo_inicial = time()
            initial=final=0
            while initial==final:
                initial=random.randint(0,round(len(G.nodes)/2))
                final=random.randint(initial,len(G.nodes)-2)
            if k<=4:
                T=nx.maximum_flow(G, initial, final)
                algorit=1
            if 4<k<=9:
                T=nx.algorithms.flow.edmonds_karp(G, initial, final)
                algorit=2
            if 9<k<=14:
                T=nx.algorithms.flow.boykov_kolmogorov(G,initial,final) 
                algorit=3
            tiempo_final = time()
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            
            print(len(G.nodes))
            matriz[aux,0]=algorit
            matriz[aux,1]=graf
            matriz[aux,2]=len(G.nodes)
            matriz[aux,3]=nx.density(G)
            matriz[aux,4]=tiempo_ejecucion
            aux+=1

data=pd.DataFrame(matriz)
data.columns=['Algoritmo', 'Generador','Orden','Densidad','Tiempo']


######################################################################################ANOVA###################################

data.columns=['Algoritmo', 'Generador','Orden','Densidad','Tiempo'] 
model_name = ols('Tiempo ~ Algoritmo+Generador+Orden+Densidad', data=data).fit()
f = open('Ols.txt', "w")
f.write('%s \t' %model_name.summary())
f.close()

aov_table = sm.stats.anova_lm(model_name, typ=2)
f = open('Anova.txt', "w")
f.write('%s \t' %aov_table)
f.close()
#----------------------------------------------------------------------------------------------turkey------------------------------------#


for i in (1,2,3):
    data_filter= data[data['Generador'] == i]
    plt.hist(data_filter['Tiempo'], histtype='bar')
    plt.show()
    
    
    
    
    
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='bwr', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
#plt.xticks(rotation=0)
ax.set_yticks(ticks)
ax.set_xticklabels(( 'Algoritmo','Generador', 'Orden', 'Densidad','Tiempo'))
ax.set_yticklabels(( 'Algoritmo','Generador', 'Orden', 'Densidad','Tiempo'))
plt.title('Matriz de Correlaciones', pad=16.0)
plt.show()
plt.savefig('Correlaciones.eps', format='eps', dpi=1000)    





for j in ('Algoritmo','Generador','Orden', 'Densidad'):  
    nombres=[]
    contador=0
    x1=[]
    fig, ax = plt.subplots()
    for i in data[j].unique():
        data_filter= data[data[j] == i]
        #print(data_filter)
        #print('1aqui')
        plt.errorbar(i,np.mean(data_filter['Tiempo']), yerr=np.std(data_filter['Tiempo']),fmt='o',color='blue',alpha=10) 
        #contador+=1
        nombres.append(str(int(i)))
        #print('2aqui')
        x1.append(int(i))
    #print(x1)
    #print('3aqui')
    #print(nombres)
    plt.xlabel(j, size=14)
    plt.ylabel('Tiempo', size=14)
    plt.title('Errorbars',size=18)
    labels = nombres
    ax.set_xticks(x1)
    ax.set_xticklabels(labels, minor=False)
    plt.show()
    plt.savefig('errors bars.eps', format='eps', dpi=1000) 


for color in ('blue', 'green', 'red'):
    for marker in ('*','v','o'):
        if color=='blue':
            aux1=1
        elif color=='green':
            aux1=2
        elif color=='red':
            aux1=3
        else:
            print('error')
        if marker=='*':
            aux2=1
        elif marker=='v':
            aux2=2
        elif marker=='o':
            aux2=3
        else:
            print('error')
        x=[i for i in range(len(matriz)) if (matriz[i,1]==aux1 and matriz[i,0]==aux2)]
        y=[matriz[i,4] for i in range(len(matriz)) if (matriz[i,1]==aux1 and matriz[i,0]==aux2)]
        plt.scatter(x, y, marker=marker, c=color)
        #plt.ylim(0,1)
        #plt.xlim(200,500)
plt.xlabel('Observaciones', size=14)
plt.ylabel('Tiempo', size=14)
plt.title('Algoritmo y Generador contra tiempo',size=18)
blue_patch = mpatches.Patch(color='blue', label='Gen1')
green_patch = mpatches.Patch(color='green', label='Gen2')
red_patch = mpatches.Patch(color='red', label='Gen3')
cuadrado_line = mlines.Line2D([], [],color='black', marker='*', markersize=10, label='Alg 1')
triangulo_line = mlines.Line2D([], [],color='black', marker='v', markersize=10, label='Alg 2')
circulo_line = mlines.Line2D([], [],color='black', marker='o', markersize=10, label='Alg 3')
plt.legend(handles=[blue_patch,green_patch,red_patch,cuadrado_line,triangulo_line,circulo_line],bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)        
plt.savefig('scater.eps', format='eps', dpi=1000)










  
    
        
        








T=nx.gnm_random_graph(27,27*27, seed=None, directed=False)  
nx.draw(T, with_labels=True, edge_color='black', pos=nx.spring_layout(T), node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')