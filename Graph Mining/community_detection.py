"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


G = nx.read_edgelist('CA-HepTh.txt', comments='#', delimiter='\t', create_using=nx.Graph())



def spectral_clustering(G, k):
    

    W = nx.adjacency_matrix(G)
    # degree matrix
    D = np.diag(np.sum(np.array(W.todense()), axis=1))
    L = D - W

    e, v =scipy.sparse.linalg.eigs(L.astype(float), k=k, which='SM')

    U = np.array(np.real(v))

    km = KMeans(init='k-means++', n_clusters=k)
    km.fit(U )

    labels=km.labels_

    idx=0
    clustering={}
    for node in G.nodes()  :
       clustering[node]=labels[idx]
       idx+=1

    ##################
    
    return clustering



############## Task 6

##################
gcc_nodes = max(nx.connected_components(G), key=len)
GCC = G.subgraph(gcc_nodes)
clustering=spectral_clustering(GCC, 50)
##################



############## Task 7
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
   ##################
    modularity=0
    m=G.number_of_edges()

    for k in range(min(clustering.values()),max(clustering.values())+1):
      d_c=0
      communitie=set()
      for node in G.nodes()  :
        
        if clustering[node] == k:
          d_c+=G.degree(node)
          communitie.add(node)
      l_c=G.subgraph(communitie).number_of_edges()
      
      modularity+=(l_c/m) - (d_c/(2*m))**2



############## Task 8

##################
random_clustering={}
for node in GCC.nodes()  :
       random_clustering[node]=randint(0,49)

print(modularity(GCC, clustering))

print(modularity(GCC, random_clustering))
##################