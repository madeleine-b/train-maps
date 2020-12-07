import numpy as np
import pandas as pd
import json
import networkx as nx 

mta_file = "mta-data/mta_adjacency_mat"
mbta_file = "mbta-data/mbta_adj_mat"

def spectra(file):
	with open(file, "r") as f:
		adj_mat = pd.read_json(f)

	adj_mat = adj_mat.to_numpy()

	G = nx.from_numpy_matrix(adj_mat) 
	assert nx.is_connected(G), "G is not connected"
	assert np.all(adj_mat.T == adj_mat), "G is not undirected"

	degree = np.diag(np.sum(adj_mat, axis=1))
	d_inv = np.linalg.inv(degree)
	random_walk = np.matmul(adj_mat, d_inv)
	assert np.allclose(np.sum(random_walk, axis=0), 1), "Random walk isn't stochastic"

	laplacian = degree - adj_mat

	omegas, w = np.linalg.eig(random_walk)
	lambdas, v = np.linalg.eig(laplacian)

	omegas = sorted(np.real(omegas), reverse=True)
	lambdas = sorted(np.real(lambdas), reverse=True)
	print(lambdas[1])

	w_pi = max(omegas[1], -omegas[-1])
	print(w_pi)

spectra(mta_file)
print("--------------------")
spectra(mbta_file)