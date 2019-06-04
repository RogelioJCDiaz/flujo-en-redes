import numpy as np
import pandas as pd
import time
import csv
import scipy as sc
import networkx as nx
import matplotlib.pyplot as plt
from time import time 
import statistics as st
import random
from datetime import datetime
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

rango=200
Grafo=nx.random_geometric_graph(rango,0.125)
G=Grafo.copy()
clustering=[]
centrality=[]
clossenes=[]
pagerank=[]
degree=[]
lista=[]
lista_hub=[]
lista_bound=[]
lista[:]=G.edges
"""A;ADIR CAPACIDADES Y PESOS A LAS ARISTAS"""
for i in range(len(lista)):
    R=np.random.normal(loc=20, scale=5.0, size=None)
    E=np.random.normal(loc=20, scale=5.0, size=None)
    G.add_edge(lista[i][0],lista[i][1],capacity=R, weight=E)
"""GUARDO CARACTERISTICAS DE LOS NODOS"""        
for i in Grafo.nodes:
        clustering.append(((nx.clustering(G, nodes=i)),i))
        clossenes.append(((nx.closeness_centrality(G, u=i)),i))
        #centrality.append(((nx.load_centrality(G, v=i)),i)) 
        pagerank.append(((nx.pagerank(G, alpha=0.9)[i]),i))
        degree.append(((nx.degree(G,i)),i))
        
data=np.arange(rango*5,dtype=float).reshape(rango,5)
cont=0
for i in (clustering,clossenes,pagerank,degree):
    auxd=np.array(i)
    data[:,cont]=auxd[:,0]
    cont+=1
        
"""SEPARAR CONJUNTO EN POSIBLES HUB Y NODOS BORDES"""          
degree=sorted(degree, reverse=True)
clustering=sorted(clustering,reverse=True)
clossenes=sorted(clossenes,reverse=True)
centrality=sorted(centrality,reverse=True)
pagerank=sorted(pagerank,reverse=True)




for i in range(int((rango*0.5)-1)):
    lista_hub.append(degree[i][1])
for i in range(int((len(degree))*0.5),len(degree)):
    lista_bound.append(degree[i][1])        
"""DEFINIR FUENTES Y SUMIDEROS"""
sumideros=[]
fuentes=[]
for i in range(len(lista_bound)):
    if int(len(lista_bound)/2)>=i:
        fuentes.append(lista_bound[i])
    else:
        sumideros.append(lista_bound[i])                
                               
"""AGREGAR AL GRAFO NODOS DE CAPACIDAD INFINITOS"""

G.add_node('sumidero')
G.add_node('fuente')
for i in range(len(fuentes)):
    G.add_edge('fuente',fuentes[i],capacity=9999999,weight=0)    
for i in range(len(sumideros)):    
    G.add_edge('sumidero',sumideros[i],capacity=9999999,weight=0)



"""CALCULAR VARIACION DE FLUJO AL ELIMINAR NODOS"""

H=G.copy()
desv=[]
T=nx.maximum_flow(G, 'fuente', 'sumidero')
for i in range(rango):             
     try:
         H.remove_node(degree[i][1])
         t=nx.maximum_flow(H,'fuente', 'sumidero')
         H=G.copy()
         desv.append((int(T[0]-t[0]),degree[i][1]))
     except:
          H=G.copy()
          desv.append((999999,degree[i][1]))
     
     
"""CALCULAR VARIACIONES DE CAMINOS MINIMOS"""

H=G.copy()
Short=0
short=0
desv1=[]
T=nx.shortest_path(G, 'fuente', 'sumidero', weight=True, method='dijkstra')
for i in range(len(T)-1):    
    Short+=H[T[i]][T[i+1]]['weight']
for i in Grafo.nodes:
     try:   
         H.remove_node(degree[i][1])
         t=nx.shortest_path(H, 'fuente', 'sumidero', weight=True, method='dijkstra')
         short=0
         for j in range(len(t)-1):    
             short+=H[t[j]][t[j+1]]['weight']         
         H=G.copy()
         desv1.append((int(short-Short),degree[i][1]))
     except:
         H=G.copy()
         desv1.append((9999999,degree[i][1]))

#desv=sorted(desv,reverse=True)
#desv1=sorted(desv1,reverse=True)    
desv=np.array(desv)
desv1=np.array(desv1)     




"""COEFICIENTE DE EXPANSION"""
conect=[]
for i in Grafo.nodes:
    K=nx.bfs_tree(Grafo,i)
    conectividad=0
    for j in list(K[i]):
        conectividad+=len(list(K[j]))
    conect.append((conectividad+len(K[i]),degree[i][1]))    


#conect=sorted(conect,reverse=True)
conect=np.array(conect)
"""INTERSECCIONES"""

#disconexo=np.array(disconexo)
conect=np.array(conect)
degree=np.array(degree)
clustering=np.array(clustering)
clossenes=np.array(clossenes)
centrality=np.array(centrality)
pagerank=np.array(pagerank)


for i in range(len(clossenes)):
    if np.percentile(clossenes[:,0], 75)>clossenes[i,0]:
        clossenes1=clossenes[0:i,:]
        break
    
for i in range(len(centrality)):
    if np.percentile(centrality[:,0],75)>centrality[i,0]:
        centrality1=centrality[0:i,:]
        break

for i in range(len(pagerank)):
    if np.percentile(pagerank[:,0],75)>pagerank[i,0]:
        pagerank1=pagerank[0:i,:]
        break

for i in range(len(clustering)):
    if np.percentile(clustering[:,0],75)>clustering[i,0]:
        clustering1=clustering[0:i,:]
        break

"""for i in range(len(disconexo)):
    if np.percentile(disconexo,75)>disconexo[i]:
        disconexo1=disconexo[0:i]
        break"""
for i in range(len(degree)):
    if np.percentile(degree[:,0],75)>degree[i,0]:
        degree1=degree[0:i,:]
        break         

for i in range(len(desv)):
        if np.percentile(desv[:,0],65)>desv[i,0]:
            desv_1=desv[0:i,:]
            break

for i in range(len(desv1)):
    if np.percentile(desv1[:,0],75)>desv1[i,0]:
        desv1_1=desv1[0:i,:]
        break

for i in range(len(conect)):
    if np.percentile(conect[:,0],75)>conect[i,0]:
        conect1=conect[0:i,:]
        break

cant=np.arange(rango*2,dtype=float).reshape(rango,2)
for i in range(len(conect[:,1])):
    cant[i,0]=conect[i,0]+desv[i,0]+desv1[i,0]
    cant[i,1]=i
    
    
    
    
"""|set(disconexo)|set(clustering[:,1])|set(pagerank[:,1])|set(centrality[:,1])|set(clossenes[:,1])|set(degree[:,1])"""
cant1=list(((set(desv_1[:,1])|set(desv1_1[:,1]))&set(conect[:,1]))) 


for i in (desv,desv1,degree,clustering,pagerank,centrality,clossenes,conect):
    
      
#for i in (degree[:,1],desv[:,1],desv1[:,1],conect[:,1]):
    pos=nx.get_node_attributes(Grafo,'pos')
    
    # find node near center (0.5,0.5)
    dmin=1
    ncenter=0 
    for n in pos:
        x,y=pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d
    
    # color by path length from node near center
    p=nx.single_source_shortest_path_length(Grafo,ncenter)
    plt.figure(figsize=(8,8))
    nx.draw_networkx_edges(Grafo,pos,nodelist=[ncenter],alpha=0.4,edgecolor='grey')
    
    nx.draw_networkx_nodes(Grafo,pos,nodelist=list(p.keys()),node_size=80,node_color=list(p.values()),cmap=plt.cm.Reds_r)
    #nx.draw_networkx_nodes(Grafo,pos,node_size=90,nodelist=cant,node_color='blue')
    #nx.draw_networkx_nodes(Grafo,pos,nodelist=list(cant[:,1]),borderWidth=200,node_size=80,node_color=list(cant[:,0]),cmap=plt.cm.Reds_r)
    #nx.draw_networkx_nodes(Grafo,pos,nodelist=cant1,node_size=80,node_color='blue')
    
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.axis('off')
#plt.savefig('inicial.eps', format='eps', dpi=1000)
plt.show()


h,a=nx.hits(Grafo,max_iter=10000)

h=[h[i] for i in range(len(h))]
a=[a[i] for i in range(len(a))]

nx.draw(h, pos=nx.kamada_kawai_layout(h), node_size=80, node_color='green', node_shape='o',with_labels=True)


data[:,4]=cant[:,0]
for i in range(len(cant)):
    if np.percentile(cant[:,0],75)<cant[i,0]:
        data[i,4]=4
    if np.percentile(cant[:,0],75)>=cant[i,0] and np.percentile(cant[:,0],50)<cant[i,0]:
        data[i,4]=3    
    if np.percentile(cant[:,0],50)>=cant[i,0] and np.percentile(cant[:,0],25)<cant[i,0]:
        data[i,4]=2    
    if np.percentile(cant[:,0],25)>=cant[i,0] and np.percentile(cant[:,0],0)<cant[i,0]:
        data[i,4]=1    





data=pd.DataFrame(data)


data.columns=['clustering','clossenes','pagerank','degree','hub']          
model_name = ols('hub~clustering+clossenes+pagerank+degree', data=data).fit()
f = open('Olst.txt', "w")
f.write('%s \t' %model_name.summary())
f.close()


lista_flujo=[]
for t in T[1].keys():
    for m in T[1][t].keys():
        if T[1][t][m]!=0:
            lista_flujo.append((t,m))








nx.draw(Grafo, with_labels=True, pos=nx.kamada_kawai_layout(Grafo), edge_color='black', node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')



"""COMPLEMETOS"""
contador=0
for i in (degree[:,0],clustering[:,0],pagerank[:,0],centrality[:,0],clossenes[:,0]):
    boxplot = plt.boxplot(list(i))
    if contador==0:
        plt.savefig('Grado.eps', format='eps', dpi=1000)
    if contador==1:
        plt.savefig('Clustering.eps', format='eps', dpi=1000)
    if contador==2:
        plt.savefig('Pagerank.eps', format='eps', dpi=1000)    
    if contador==3:
        plt.savefig('Centrality.eps', format='eps', dpi=1000)
    if contador==4:
        plt.savefig('Clossenes.eps', format='eps', dpi=1000)
    contador+=1
    plt.show()


"""ARBOL DE EXPANSION MINIMA"""

desv2=[]
H=G.copy()
J=nx.minimum_spanning_tree(G, weight='weight')
edgesJ = J.edges(data=True)
Short = sum([e[2]["weight"] for e in edgesJ])
for i in range(len(degree)):          
     H.remove_node(degree[i][1])
     j=nx.minimum_spanning_tree(H, weight='weight')
     edgesJ = j.edges(data=True)
     short = sum([e[2]["weight"] for e in edgesJ])
     H=G.copy()
     print()
     desv2.append((Short-short,degree[i][1])) 
desv2=sorted(desv2,reverse=True)















G.add_edge(1, 2)
G[77][98].update({'nada': 5})



G.add_edge(1, 2)
G.edges[1, 2].update({1: 4})




















        
for i in (clustering,clossenes,centrality,eccentrity,pagerank,degree):
    normed=(i-st.mean(i))/st.stdev(i)
    plt.hist(i)
    plt.show()
    print(sc.stats.kstest(normed, 'norm'))
            
        
        
G.remove_node('sumidero')               
G.remove_node('fuente')  
pos=nx.kamada_kawai_layout(G)      
nx.draw(G,pos, node_size=300, node_shape='o',with_labels=True)        
nx.draw_networkx_edges(G, pos, edgelist=None, edge_color='k')
nx.draw_networkx_edges(G, pos, edgelist=s,edge_color='blue')




        
    for w in range(ordenes[i]):
        initial=final=0        
        while initial==final:
            initial=random.randint(0,round(len(G.nodes)/2))
            final=random.randint(initial,len(G.nodes)-2)
            
        



