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




ordenes=[10,20,30,40,50]
initial=0
final=0
contador=0
data=np.arange(sum(ordenes)*13,dtype=float).reshape(sum(ordenes),13)
tiempo_inicial=0
tiempo_final =0 
tiempo_ejecucion=0  
for i in range(5):
    rango=random.randint(ordenes[i],ordenes[i]*2)
    #G=nx.dense_gnm_random_graph(ordenes[i],rango)
    G=nx.watts_strogatz_graph(ordenes[i], int(ordenes[i]/2) , 0.33 , seed=None)
    lista=[]
    lista[:]=G.edges
    width=np.arange(len(lista)*1,dtype=float).reshape(len(lista),1)
    for r in range(len(lista)):
        R=np.random.normal(loc=20, scale=5.0, size=None)
        width[r]=R
        G.add_edge(lista[r][0],lista[r][1],capacity=R)
    for w in range(ordenes[i]):
        initial=final=0        
        while initial==final:
            initial=random.randint(0,round(len(G.nodes)/2))
            final=random.randint(initial,len(G.nodes)-2)
            
        
        tiempo_inicial=time()
        T=nx.maximum_flow(G, initial, final)
        tiempo_final =time()
        tiempo_ejecucion=tiempo_final- tiempo_inicial
        
        data[contador,2]=nx.clustering(G, nodes=initial)
        data[contador,3]=nx.load_centrality(G, v=initial)
        data[contador,4]=nx.closeness_centrality(G, u=initial)
        data[contador,5]=nx.eccentricity(G, v=initial)
        data[contador,6]=nx.pagerank(G, alpha=0.9)[initial]
#-------------------------------------------------------------------------------------------------------------#
        data[contador,7]=nx.clustering(G, nodes=final)
        data[contador,8]=nx.load_centrality(G, v=final)
        data[contador,9]=nx.closeness_centrality(G, u=final)
        data[contador,10]=nx.eccentricity(G, v=final)
        data[contador,11]=nx.pagerank(G, alpha=0.9)[final]
#--------------------------------------------------------------------------------------------------------------#
        data[contador,0]=T[0]
        data[contador,1]=tiempo_ejecucion
        data[contador,12]=ordenes[i]
        contador+=1
        lista_flujo=[]
        for t in T[1].keys():
            for m in T[1][t].keys():
                if T[1][t][m]!=0:
                    lista_flujo.append((t,m))
        edges = G.edges()
        weights = [(G[u][v]['capacity'])/15 for u,v in edges]     
        nodos=[]
        nodos[:]=G.nodes
        
        nodos.remove(initial)
        nodos.remove(final)    
        cont=0
        y=1
        pos={}
        for t in range(len(G.nodes)):
            pos[t]=((t%5)*5,y)
            cont+=1
            if cont%4==0:
                y+=1
                cont=0            
        nx.draw_networkx_nodes(G, pos, [initial], node_size=300, node_color='green', node_shape='o',with_labels=True)
        nx.draw_networkx_nodes(G, pos, [final], node_size=300, node_color='r', node_shape='^',with_labels=True)    
        nx.draw_networkx_nodes(G, pos,nodos, node_size=300, node_color='grey', node_shape='o',with_labels=True)    
        nx.draw_networkx_edges(G, pos, edgelist=None, width=weights, edge_color='k', style='solid',with_labels=True)
        nx.draw_networkx_edges(G, pos, edgelist=lista_flujo, width=weights, edge_color='blue', style='solid',with_labels=True)
        plt.axis('off') 
        plt.show()
        
      
        
        
        
        
        
        
        
        
        
data=data.copy()
data=pd.DataFrame(data)
data.columns=['FO','Tiempo','clustering_fuente','load_fuente','closeness_fuente','eccentricity_fuente','Prank_fuente','clustering_sumidero','load_sumidero','closeness_sumidero','eccentricity_sumidero','Prank_sumidero','orden']        
        
        
        
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='bwr', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
#plt.xticks(rotation=0)
ax.set_yticks(ticks)
plt.title('Matriz de Correlaciones', pad=16.0)
plt.savefig('Correlaciones.eps', format='eps', dpi=1000)           
plt.show()      
        
       
data.columns=['FO','Tiempo','clustering_fuente','load_fuente','closeness_fuente','eccentricity_fuente','Prank_fuente','clustering_sumidero','load_sumidero','closeness_sumidero','eccentricity_sumidero','Prank_sumidero','orden'] 
model_name = ols('FO ~clustering_fuente+load_fuente+closeness_fuente+eccentricity_fuente+Prank_fuente+clustering_sumidero+load_sumidero+closeness_sumidero+eccentricity_sumidero+Prank_sumidero', data=data).fit()
f = open('Ols.txt', "w")
f.write('%s \t' %model_name.summary())
f.close()

aov_table = sm.stats.anova_lm(model_name, typ=2)
f = open('Anova.txt', "w")
f.write('%s \t' %aov_table)
f.close()        
        
        
data.columns=['FO','Tiempo','clustering_fuente','load_fuente','closeness_fuente','eccentricity_fuente','Prank_fuente','clustering_sumidero','load_sumidero','closeness_sumidero','eccentricity_sumidero','Prank_sumidero','orden'] 
model_name = ols('Tiempo~clustering_fuente+load_fuente+closeness_fuente+eccentricity_fuente+Prank_fuente+clustering_sumidero+load_sumidero+closeness_sumidero+eccentricity_sumidero+Prank_sumidero', data=data).fit()
f = open('Olst.txt', "w")
f.write('%s \t' %model_name.summary())
f.close()       



for d in range()
    
        
    


plt.boxplot()

        
boxplot = data.boxplot(column=['clustering_fuente','clustering_sumidero'])
plt.savefig('clustering.eps', format='eps', dpi=1000)
plt.show()
boxplot = data.boxplot(column=['load_fuente','load_sumidero'])
plt.savefig('load.eps', format='eps', dpi=1000)
plt.show()
boxplot = data.boxplot(column=['closeness_fuente','closeness_sumidero'])
plt.savefig('closeness.eps', format='eps', dpi=1000)
plt.show()
boxplot = data.boxplot(column=['eccentricity_fuente','eccentricity_sumidero'])
plt.savefig('eccentrity.eps', format='eps', dpi=1000)
plt.show()
boxplot = data.boxplot(column=['Prank_fuente','Prank_sumidero'])
plt.savefig('prank.eps', format='eps', dpi=1000)
plt.show()
boxplot = data.boxplot(column=['FO'])
plt.savefig('fun.eps', format='eps', dpi=1000)
plt.show()
boxplot = data.boxplot(column=['Tiempo'])
plt.savefig('tiempo.eps', format='eps', dpi=1000)
plt.show()









    

