# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

data = pd.read_csv('../data/edges_with_weight/edges_with_weight.csv', ';')

print(data)

# Build your graph. Note that we use the DiGraph function to create the graph!
G = nx.from_pandas_edgelist(data, 'src', 'dest', create_using=nx.DiGraph(), edge_attr='weight')

#plt.figure(1, figsize=(10,10))

# Make the graph
#nx.draw(G, with_labels=False, node_size=100, alpha=0.6, arrows=True, font_size=8, pos=nx.kamada_kawai_layout(G))

#plt.show()

G = nx.from_pandas_edgelist(data, 'src', 'dest', create_using=nx.Graph(), edge_attr='weight')
print nx.number_connected_components(G)