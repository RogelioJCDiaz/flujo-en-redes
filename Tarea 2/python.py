import networkx as nx
import matplotlib.pyplot as plt


####################################################primero''''''''''#############################################################

G=nx.Graph()
G.add_edges_from([(1,2), (1,3), (3,4), (3,5), (2,6), (2,7), (5,8), (4,10), (6,11), (7,12)]) 
nx.draw(G, with_labels=True, edge_color='black', pos=nx.spring_layout(G, ), node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')
plt.savefig('primerospring.eps',format='eps',dpi=1000)


####################################################segundo############################################################

G=nx.Graph()
G.add_edges_from([(1,2), (1,1), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,1) ]) 
nx.draw(G, with_labels=True, pos=nx.circular_layout(G), edge_color='black', node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')
G.number_of_nodes() 
plt.savefig('segundocircular.eps', format='eps', dpi=1000)

#####################################################tercero###############################################################

G=nx.Graph()

G.add_edges_from([(1,2), (1,1), (2,3), (3,4), (4,1), (1,3), (2,4)]) 
node1 = {1,2}
node2 = {3,4}
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, with_labels=True,node_size=400,nodelist=node1, node_color='r', node_shape='o')
nx.draw_networkx_nodes(G, pos, with_labels=True,node_size=400,nodelist=node2,node_color='grey', node_shape='o')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G,pos)
plt.axis('off')
plt.savefig('tercerospring.eps', format='eps', dpi=1000)

########################################################cuarto###################################################################

G=nx.DiGraph()
G.add_edges_from([(1,2), (2,3), (2,4), (2,5), (5,6), (5,7)]) 
nx.draw(G, with_labels=True, pos=nx.random_layout(G), edge_color='black', node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')
G.number_of_nodes() 
plt.savefig('cuartorandom.eps', format='eps', dpi=1000)

#######################################################quinto####################################################################


G=nx.DiGraph()
G.add_edges_from([(1,2), (2,3), (3,4), (4,1), (1,3)]) 
nx.draw(G, with_labels=True, pos=nx.kamada_kawai_layout(G), edge_color='black', node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')
G.number_of_nodes() 
plt.savefig('quintokamada.eps', format='eps', dpi=1000)


#########################################################sexto##########################################################################

G=nx.DiGraph()
G.add_edges_from([(1,2), (2,3), (3,4)]) 
node1 = {1,2}
node2 = {3,4}
pos = {1:(200, 350), 2:(550,350), 3:(650, 220), 4:(400,100), 5:(150,220)}
nodes=nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node1,node_size=400, node_color='r', node_shape='o')
nodes=nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node2,node_size=400, node_color='grey', node_shape='o')
nodes=nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G,pos)
plt.axis('off')
plt.savefig('sexto.eps', format='eps', dpi=1000)



#########################################################septimo########################################################################

G=nx.MultiGraph()
G.add_edges_from([(1,2), (2,1), (2,3,), (3,2,), (3,4,)]) 
node1 = {1,2}
node2 = {3,4}
edge1 = {(1,2)}
edge2 = {(2,1)}
edge3 = {(2,3)}
edge4 = {(3,2)}
edge5 = {(3,4)}
pos = {1:(200, 350), 2:(550,350), 3:(650, 220), 4:(400,100), 5:(150,220)}
nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node1,node_size=400, node_color='r', node_shape='o')
nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node2,node_size=400, node_color='grey', node_shape='o')
nx.draw_networkx_edges(G, pos, edgelist=edge1, edge_color='green', width=10)
nx.draw_networkx_edges(G, pos, edgelist=edge2, edge_color='red',width=5)
nx.draw_networkx_edges(G, pos, edgelist=edge3, edge_color='green',width=10)
nx.draw_networkx_edges(G, pos, edgelist=edge4, edge_color='red',width=5)
nx.draw_networkx_edges(G, pos, edgelist=edge5, edge_color='k',width=5.0)
nx.draw_networkx_labels(G,pos)
plt.axis('off') 
plt.savefig('septimo.eps', format='eps', dpi=1000)


#########################################################octavo######################################################################

R = nx.bipartite.gnmk_random_graph(3,5,10,seed=123)
edge1 = {(1,5)}
edge2 = {(5,1)}
top = nx.bipartite.sets(R)[0]
pos = nx.bipartite_layout(R, top)
nx.draw(R,pos,with_labels=True)
nx.draw_networkx_edges(R, pos, edgelist=edge1, edge_color='green', width=10)
nx.draw_networkx_edges(R, pos, edgelist=edge2, edge_color='red',width=5)
plt.axis('off')
plt.savefig('octavob.eps', format='eps', dpi=1000)

#######################################################noveno#########################################################################

G=nx.MultiGraph()
G.add_edges_from([(1,2), (2,1), (2,3,), (3,2,), (3,4,), (4,1), (1,4)]) 
node1 = {1,2}
node2 = {3,4}
pos=nx.spectral_layout(G)
nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node1,node_size=400, node_color='r', node_shape='o')
nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node2,node_size=400, node_color='grey', node_shape='o')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G,pos)


plt.axis('off')
plt.savefig('novenospectral.eps', format='eps', dpi=1000)

####################################################decimo################################################################################

G=nx.MultiDiGraph()
G.add_edges_from([(1,2), (2,1), (2,3,), (3,2,), (3,4,)]) 
node1 = {1,2}
node2 = {3,4}
pos = {1:(200, 350), 2:(550,350), 3:(650, 220), 4:(400,100), 5:(150,220)}
#nodes=nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node1,node_size=400, node_color='r', node_shape='o')
#nodes=nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node2,node_size=400, node_color='grey', node_shape='o')
#nodes=nx.draw_networkx_edges(G, pos)
nx.draw(G, with_labels=True, pos=nx.spring_layout(G), edge_color='black', node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')

plt.savefig('decimo.eps', format='eps', dpi=1000)

####################################################onceavo#############################################################################

G=nx.MultiDiGraph()
G.add_edges_from([(1,2), (2,1), (2,3), (3,2), (3,4), (4,3), (4,1), (1,4)]) 
nx.draw(G, with_labels=True, pos=nx.spring_layout(G), edge_color='black', node_color='gray',node_size=1000, edgecolors='black', font_weight='bold')
G.number_of_nodes() 
plt.savefig('grafoNAS.eps', format='eps', dpi=1000)


####################################################doceavo##############################################################################

G=nx.MultiDiGraph()
G.add_edges_from([(1,2), (2,1), (2,3), (3,2), (3,4), (4,3), (4,1), (1,4)]) 
node1 = {1,2}
node2 = {3,4}
pos = {1:(200, 350), 2:(550,350), 3:(650, 220), 4:(400,100), 5:(150,220)}
#nodes=nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node1,node_size=400, node_color='r', node_shape='o')
#nodes=nx.draw_networkx_nodes(G, pos, with_labels=True, nodelist=node2,node_size=400, node_color='grey', node_shape='o')
#nodes=nx.draw_networkx_edges(G, pos)
pos = nx.nx_agraph.graphviz_layout(G)
pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
#draw_spectral(G)
#generic_graph_view(G, create_using=None)
#plt.savefig('grafoNAS.eps', format='eps', dpi=1000)