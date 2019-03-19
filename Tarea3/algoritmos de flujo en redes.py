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

random.seed(datetime.now())
guardar_n_e={}
guardar_n_e['nodos']=[]
guardar_n_e['edges']=[]
guardar_n_e['media']=[]
guardar_n_e['desv']=[]
#-------------------------------------------------------------------Shortest_path---------------------------------------------------------# 

df=pd.read_csv('instancia36.txt',sep=None,names=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35))
matrizadd=np.matrix(df)
Graf=nx.Graph()
rog=0
diccionario_inst_Shortest_path={}
while rog<=4:
    Graf.clear()
    rango=random.randint(random.randint(10,len(matrizadd)),len(matrizadd))
    for i in range(rango):
        for j in range(rango):
            if matrizadd[i,j]!=0:
                matrizadd[j,i]=0
                #Graf.add_weighted_edges_from([(i,j,matrizadd[i,j])])
                Graf.add_edges_from([(i,j)])
    cont=0
    lista_tiempos=[] 
    lista_tiempos_completos={}    
    tiempo_de_paro=0
    tiempo_inicial=0
    tiempo_final =0 
    tiempo_ejecucion=0    
    for r in range(30):
        lista_tiempos_completos[r+1]=[]
        tiempo_de_paro=0    
        while tiempo_de_paro<1:            
            tiempo_inicial = time()             
            nx.shortest_path(Graf, source=None, target=None, weight=None, method='dijkstra')           
            tiempo_final = time()  
            tiempo_ejecucion = tiempo_final - tiempo_inicial        
            if tiempo_ejecucion>0.0:
                lista_tiempos_completos[r+1].append((tiempo_ejecucion*10000))  
            tiempo_de_paro+=tiempo_ejecucion  
    guardar_n_e[rog]=[]
    diccionario_inst_Shortest_path[rog]=[]
    for i in lista_tiempos_completos.keys():
         media=np.mean(lista_tiempos_completos[i])
         diccionario_inst_Shortest_path[rog].append(media)
    guardar_n_e['nodos'].append(len(Graf.nodes))
    guardar_n_e['edges'].append(len(Graf.edges))  
    guardar_n_e['media'].append(np.mean(diccionario_inst_Shortest_path[rog]))
    guardar_n_e['desv'].append(np.std(diccionario_inst_Shortest_path[rog]))
    rog+=1           

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.hist(diccionario_inst_Shortest_path[0],  histtype='bar')
ax0.set_title('Tiempo promedio instancia 1')
ax1.hist(diccionario_inst_Shortest_path[1], histtype='bar')
ax1.set_title('Tiempo promedio instancia 2')
ax2.hist(diccionario_inst_Shortest_path[2], histtype='bar')
ax2.set_title('Tiempo promedio instancia 3')
ax3.hist(diccionario_inst_Shortest_path[3],  histtype='bar')
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('Shortest_path.eps', format='eps', dpi=1000)
plt.show()    




fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.violinplot(diccionario_inst_Shortest_path[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.violinplot(diccionario_inst_Shortest_path[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.violinplot(diccionario_inst_Shortest_path[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.violinplot(diccionario_inst_Shortest_path[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('violintminimum_Shortest_path.eps', format='eps', dpi=1000)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.boxplot(diccionario_inst_Shortest_path[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.boxplot(diccionario_inst_Shortest_path[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.boxplot(diccionario_inst_Shortest_path[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.boxplot(diccionario_inst_Shortest_path[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('boxplotminimum_Shortest_path.eps', format='eps', dpi=1000)
plt.show()  





        
#--------------------------------------------------------------------------------dfs_tree--------------------------------------------#    
df=pd.read_csv('instancia36.txt',sep=None,names=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35))
matrizadd=np.matrix(df)
Graf=nx.Graph()
rog=0
diccionario_inst_dfs_tree={}
while rog<=4:
    
    
    Graf.clear()
    rango=random.randint(random.randint(10,len(matrizadd)),len(matrizadd))
    
    for i in range(rango):
        for j in range(rango):
            if matrizadd[i,j]!=0:
                matrizadd[j,i]=0
                #Graf.add_weighted_edges_from([(i,j,matrizadd[i,j])])
                Graf.add_edges_from([(i,j)])
    cont=0
    lista_tiempos=[] 
    lista_tiempos_completos={}    
    tiempo_de_paro=0
    tiempo_inicial=0
    tiempo_final =0 
    tiempo_ejecucion=0
        
    for r in range(30):
        lista_tiempos_completos[r+1]=[]
        tiempo_de_paro=0    
        while tiempo_de_paro<1:            
            tiempo_inicial = time()             
            T=nx.dfs_tree(Graf, source=0, depth_limit=None)        
            tiempo_final = time()  
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            
            if tiempo_ejecucion>0.0:
                lista_tiempos_completos[r+1].append((tiempo_ejecucion*10000))  
            tiempo_de_paro+=tiempo_ejecucion  
               
   
    #diccionario_inst[rog]=[]
    diccionario_inst_dfs_tree[rog]=[]
    for i in lista_tiempos_completos.keys():
         
         #desviacion=statistics.stdev(lista_tiempos_completos[i]) 
         
         media=np.mean(lista_tiempos_completos[i])
         diccionario_inst_dfs_tree[rog].append(media)
    
    guardar_n_e['nodos'].append(len(Graf.nodes))
    guardar_n_e['edges'].append(len(Graf.edges))  
    guardar_n_e['media'].append(np.mean(diccionario_inst_dfs_tree[rog])) 
    guardar_n_e['desv'].append(np.std(diccionario_inst_dfs_tree[rog]))
    rog+=1  
     
fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.hist(diccionario_inst_dfs_tree[0],  histtype='bar')
ax0.set_title('Tiempo promedio instancia 1')
ax1.hist(diccionario_inst_dfs_tree[1], histtype='bar')
ax1.set_title('Tiempo promedio instancia 2')
ax2.hist(diccionario_inst_dfs_tree[2], histtype='bar')
ax2.set_title('Tiempo promedio instancia 3')
ax3.hist(diccionario_inst_dfs_tree[3],  histtype='bar')
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('dfs_tre.eps', format='eps', dpi=1000)
plt.show() 




fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.violinplot(diccionario_inst_dfs_tree[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.violinplot(diccionario_inst_dfs_tree[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.violinplot(diccionario_inst_dfs_tree[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.violinplot(diccionario_inst_dfs_tree[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('violintminimum_dfs_tree.eps', format='eps', dpi=1000)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.boxplot(diccionario_inst_dfs_tree[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.boxplot(diccionario_inst_dfs_tree[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.boxplot(diccionario_inst_dfs_tree[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.boxplot(diccionario_inst_dfs_tree[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('boxplotminimum_dfs_tree.eps', format='eps', dpi=1000)
plt.show()    
    

#----------------------------------------------------------------------------------greedy_color-----------------------------------------------------------------

df=pd.read_csv('instancia36.txt',sep=None,names=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35))
matrizadd=np.matrix(df)
Graf=nx.DiGraph()
rog=0

diccionario_inst_greedy_color={}

while rog<=4:
    
    Graf.clear()
    rango=random.randint(random.randint(10,len(matrizadd)),len(matrizadd))
        
    
    for i in range(rango):
        for j in range(rango):
            if matrizadd[i,j]!=0:
                matrizadd[j,i]=0
                Graf.add_edges_from([(i,j)])
                
          
    cont=0
    lista_tiempos=[] 
    lista_tiempos_completos={}    
    tiempo_de_paro=0
    tiempo_inicial=0
    tiempo_final =0 
    tiempo_ejecucion=0
        
    for r in range(30):
        lista_tiempos_completos[r+1]=[]
        tiempo_de_paro=0    
        while tiempo_de_paro<1:            
            tiempo_inicial = time()             
            Dic=nx.greedy_color(Graf, strategy='largest_first', interchange=False)           
            tiempo_final = time()  
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            
            if tiempo_ejecucion>0.0:
                lista_tiempos_completos[r+1].append((tiempo_ejecucion*10000))  
            tiempo_de_paro+=tiempo_ejecucion  
               
   
    #diccionario_inst[rog]=[]
    diccionario_inst_greedy_color[rog]=[]

    for i in lista_tiempos_completos.keys():
         
         #desviacion=statistics.stdev(lista_tiempos_completos[i]) 
         
         media=np.mean(lista_tiempos_completos[i])
         diccionario_inst_greedy_color[rog].append(media) 
         
    guardar_n_e['nodos'].append(len(Graf.nodes))
    guardar_n_e['edges'].append(len(Graf.edges))  
    guardar_n_e['media'].append(np.mean(diccionario_inst_greedy_color[rog]))
    guardar_n_e['desv'].append(np.std(diccionario_inst_greedy_color[rog]))
    rog+=1 



fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.hist(diccionario_inst_greedy_color[0],  histtype='bar')
ax0.set_title('Tiempo promedio instancia 1')
ax1.hist(diccionario_inst_greedy_color[1], histtype='bar')
ax1.set_title('Tiempo promedio instancia 2')
ax2.hist(diccionario_inst_greedy_color[2], histtype='bar')
ax2.set_title('Tiempo promedio instancia 3')
ax3.hist(diccionario_inst_greedy_color[3],  histtype='bar')
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('greedy_color.eps', format='eps', dpi=1000)
plt.show()  



fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.violinplot(diccionario_inst_greedy_color[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.violinplot(diccionario_inst_greedy_color[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.violinplot(diccionario_inst_greedy_color[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.violinplot(diccionario_inst_greedy_color[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('violintminimum_greedy_color.eps', format='eps', dpi=1000)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.boxplot(diccionario_inst_greedy_color[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.boxplot(diccionario_inst_greedy_color[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.boxplot(diccionario_inst_greedy_color[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.boxplot(diccionario_inst_greedy_color[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('boxplotminimum_greedy_colo.eps', format='eps', dpi=1000)
plt.show()  
   

  
#----------------------------------------------------------------------------------maximun_flow------------------------------------------------------------------
   
df=pd.read_csv('instancia36.txt',sep=None,names=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35))
matrizadd=np.matrix(df)
Graf=nx.DiGraph()
rog=0
diccionario_inst_maximun_flow={}
while rog<=4:
    
    
    Graf.clear()
    rango=random.randint(random.randint(10,len(matrizadd)),len(matrizadd))
    
    for i in range(rango):
        for j in range(rango):
            if matrizadd[i,j]!=0:
                matrizadd[j,i]=0
                Graf.add_weighted_edges_from([(i,j,matrizadd[i,j])])
                #Graf.add_edges_from([(i,j)])
    cont=0
    lista_tiempos=[] 
    lista_tiempos_completos={}    
    tiempo_de_paro=0
    tiempo_inicial=0
    tiempo_final =0 
    tiempo_ejecucion=0
        
    for r in range(30):
        lista_tiempos_completos[r+1]=[]
        tiempo_de_paro=0    
        while tiempo_de_paro<1:            
            tiempo_inicial = time()             
            Dic2=nx.maximum_flow(Graf, 1 ,(rango-2), capacity='weight', flow_func=None)        
            tiempo_final = time()  
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            
            if tiempo_ejecucion>0.0:
                lista_tiempos_completos[r+1].append((tiempo_ejecucion*10000))  
            tiempo_de_paro+=tiempo_ejecucion  
               
   
    #diccionario_inst[rog]=[]
    diccionario_inst_maximun_flow[rog]=[]
    for i in lista_tiempos_completos.keys():
         
         #desviacion=statistics.stdev(lista_tiempos_completos[i]) 
         
         media=np.mean(lista_tiempos_completos[i])
         diccionario_inst_maximun_flow[rog].append(media) 
    
    guardar_n_e['nodos'].append(len(Graf.nodes))
    guardar_n_e['edges'].append(len(Graf.edges))  
    guardar_n_e['media'].append(np.mean(diccionario_inst_maximun_flow[rog]))
    guardar_n_e['desv'].append(np.std(diccionario_inst_maximun_flow[rog]))
    rog+=1

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.hist(diccionario_inst_maximun_flow[0],  histtype='bar')
ax0.set_title('Tiempo promedio instancia 1')
ax1.hist(diccionario_inst_maximun_flow[1], histtype='bar')
ax1.set_title('Tiempo promedio instancia 2')
ax2.hist(diccionario_inst_maximun_flow[2], histtype='bar')
ax2.set_title('Tiempo promedio instancia 3')
ax3.hist(diccionario_inst_maximun_flow[3],  histtype='bar')
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('maximun_flow.eps', format='eps', dpi=1000)
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.violinplot(diccionario_inst_maximun_flow[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.violinplot(diccionario_inst_maximun_flow[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.violinplot(diccionario_inst_maximun_flow[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.violinplot(diccionario_inst_maximun_flow[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('violintminimum_maximun_flowe.eps', format='eps', dpi=1000)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.boxplot(diccionario_inst_maximun_flow[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.boxplot(diccionario_inst_maximun_flow[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.boxplot(diccionario_inst_maximun_flow[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.boxplot(diccionario_inst_maximun_flow[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('boxplotminimum_maximun_flowe.eps', format='eps', dpi=1000)
plt.show()





#-------------------------------------------------------------minimum_spanning_tree-----------------------------------------#

df=pd.read_csv('instancia36.txt',sep=None,names=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35))
matrizadd=np.matrix(df)
Graf=nx.Graph()
rog=0
diccionario_inst_minimum_spanning_tree={}
while rog<=4:
    
    
    Graf.clear()
    rango=random.randint(random.randint(10,len(matrizadd)),len(matrizadd))
    
    for i in range(rango):
        for j in range(rango):
            if matrizadd[i,j]!=0:
                matrizadd[j,i]=0
                #Graf.add_weighted_edges_from([(i,j,matrizadd[i,j])])
                Graf.add_edges_from([(i,j)])
    cont=0
    lista_tiempos=[] 
    lista_tiempos_completos={}    
    tiempo_de_paro=0
    tiempo_inicial=0
    tiempo_final =0 
    tiempo_ejecucion=0
        
    for r in range(30):
        lista_tiempos_completos[r+1]=[]
        tiempo_de_paro=0    
        while tiempo_de_paro<1:            
            tiempo_inicial = time()             
            R=nx.minimum_spanning_tree(Graf, weight='weight')         
            tiempo_final = time()  
            tiempo_ejecucion = tiempo_final - tiempo_inicial
            
            if tiempo_ejecucion>0.0:
                lista_tiempos_completos[r+1].append((tiempo_ejecucion*10000))  
            tiempo_de_paro+=tiempo_ejecucion  
               
   
    #diccionario_inst[rog]=[]
    diccionario_inst_minimum_spanning_tree[rog]=[]
    for i in lista_tiempos_completos.keys():
         
         #desviacion=statistics.stdev(lista_tiempos_completos[i]) 
         
         media=np.mean(lista_tiempos_completos[i])
         diccionario_inst_minimum_spanning_tree[rog].append(media) 
         
    guardar_n_e['nodos'].append(len(Graf.nodes))
    guardar_n_e['edges'].append(len(Graf.edges))  
    guardar_n_e['media'].append(np.mean(diccionario_inst_minimum_spanning_tree[rog])) 
    guardar_n_e['desv'].append(np.std(diccionario_inst_minimum_spanning_tree[rog]))    
    rog+=1

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.hist(diccionario_inst_minimum_spanning_tree[0],  histtype='bar')
ax0.set_title('Tiempo promedio instancia 1')
ax1.hist(diccionario_inst_minimum_spanning_tree[1], histtype='bar')
ax1.set_title('Tiempo promedio instancia 2')
ax2.hist(diccionario_inst_minimum_spanning_tree[2], histtype='bar')
ax2.set_title('Tiempo promedio instancia 3')
ax3.hist(diccionario_inst_minimum_spanning_tree[3],  histtype='bar')
ax3.set_title('Tiempo promedio instancia 4')

fig.tight_layout()
plt.savefig('minimum_spanning_tree.eps', format='eps', dpi=1000)
plt.show()



fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.violinplot(diccionario_inst_minimum_spanning_tree[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.violinplot(diccionario_inst_minimum_spanning_tree[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.violinplot(diccionario_inst_minimum_spanning_tree[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.violinplot(diccionario_inst_minimum_spanning_tree[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('violintminimum_spanning_tree.eps', format='eps', dpi=1000)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3= axes.flatten()
ax0.boxplot(diccionario_inst_minimum_spanning_tree[0])
ax0.set_title('Tiempo promedio instancia 1')
ax1.boxplot(diccionario_inst_minimum_spanning_tree[1])
ax1.set_title('Tiempo promedio instancia 2')
ax2.boxplot(diccionario_inst_minimum_spanning_tree[2])
ax2.set_title('Tiempo promedio instancia 3')
ax3.boxplot(diccionario_inst_minimum_spanning_tree[3])
ax3.set_title('Tiempo promedio instancia 4')
fig.tight_layout()
plt.savefig('boxplotminimum_spanning_tree.eps', format='eps', dpi=1000)
plt.show()



























nodos=[]
edges=[]
medias=[]
desv=[]

for i in range(25):
    nodos.append(guardart[i][0])
    edges.append(guardart[i][1])
    medias.append(guardart[i][3])
    desv.append(guardart[i][2])


for i in range(25):
    if(i<=4):
        plt.errorbar(guardar_n_e['media'][i],guardar_n_e['edges'][i], xerr=guardar_n_e['desv'][i], fmt='+',color='k',alpha=1)
    if(4<i<=9):
        plt.errorbar(guardar_n_e['media'][i],guardar_n_e['edges'][i], xerr=guardar_n_e['desv'][i], fmt='o',color='orange',alpha=1)
    if(9<i<=14):
        plt.errorbar(guardar_n_e['media'][i],guardar_n_e['edges'][i], xerr=guardar_n_e['desv'][i],fmt='^',color='green',alpha=1)
    if(14<i<=19):
        plt.errorbar(guardar_n_e['media'][i],guardar_n_e['edges'][i], xerr=guardar_n_e['desv'][i], fmt='>',color='r',alpha=1)
    if(19<i<=24):
        plt.errorbar(guardar_n_e['media'][i],guardar_n_e['edges'][i], xerr=guardar_n_e['desv'][i], fmt='<',color='b',alpha=1)
#plt.xlim((9.97,9.99))            
plt.savefig('errorsbarsedges.eps', format='eps', dpi=1000)






# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
plt.errorbar(guardar_n_e['media'],guardar_n_e['nodos'], xerr=guardar_n_e['desv'],fmt='o')
#plt.ylim((None, 40))
plt.xlim((9.97,9.99))
plt.savefig('errorsbars.eps', format='eps', dpi=1000)
plt.title("Errorbarsnodos1")











































 
    #nx.draw(T, with_labels=True, edge_color='black', pos=nx.spring_layout(T), node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')

    """listarojo=[]
    listaazul=[]
    listagris=[]
        
    for t in range(len(matrizadd)):
        if Dic[t]==0:
             listarojo.append(t)
        if Dic[t]==1:
             listaazul.append(t)
        if Dic[t]==2:
             listagris.append(t)
    
    pos=nx.kamada_kawai_layout(Graf)
    nx.draw_networkx_nodes(Graf, pos, with_labels=True,node_size=400,nodelist=listarojo, node_color='r', node_shape='o')
    nx.draw_networkx_nodes(Graf, pos, with_labels=True,node_size=400,nodelist=listaazul,node_color='blue', node_shape='o')
    nx.draw_networkx_nodes(Graf, pos, with_labels=True,node_size=400,nodelist=listagris, node_color='grey', node_shape='o')
    nx.draw_networkx_edges(Graf, pos)
    nx.draw_networkx_labels(Graf,pos)
    plt.axis('off')
    
    listasr=lista_tiempos_completos[1]"""