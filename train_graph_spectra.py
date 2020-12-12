import numpy as np
import pandas as pd
import json
import networkx as nx 
import matplotlib.pyplot as plt

mta_file = "mta-data/mta_adjacency_mat"
mbta_file = "mbta-data/mbta_adj_mat"

def spectra(file):
	print(file)
	with open(file, "r") as f:
		adj_mat = pd.read_json(f)

	adj_mat = adj_mat.to_numpy()

	G = nx.from_numpy_matrix(adj_mat) 
	assert nx.is_connected(G), "G is not connected"
	assert np.all(adj_mat.T == adj_mat), "G is not undirected"

	degree = np.diag(np.sum(adj_mat, axis=1))
	print("n:", len(G.nodes))
	print("max degree:", np.max(np.sum(adj_mat, axis=1)))
	print("min degree:", np.min(np.sum(adj_mat, axis=1)))
	d_inv = np.linalg.inv(degree)
	random_walk = np.matmul(adj_mat, d_inv)
	assert np.allclose(np.sum(random_walk, axis=0), 1), "Random walk isn't stochastic"

	laplacian = degree - adj_mat
	normalized_laplacian = np.matmul(np.matmul(d_inv, laplacian), d_inv)

	omegas, w = np.linalg.eig(random_walk)
	lambdas, v = np.linalg.eig(laplacian)
	nus, v2 = np.linalg.eig(normalized_laplacian)

	omegas = sorted(np.real(omegas), reverse=True)
	lambdas = sorted(np.real(lambdas))
	nus = sorted(np.real(nus))
	print("lambda_2", lambdas[1])
	print("nu_2", nus[1])

	w_pi = max(omegas[1], -omegas[-1])
	print("w_pi", w_pi)


def perf_metrics():
	with open("mbta-data/perfmetrics_rail.csv", "r") as f:
		metrics18_19 = pd.read_csv(f)
		wait_time_avg = metrics18_19["otp_numerator"].to_numpy().sum() / metrics18_19["otp_denominator"].to_numpy().sum()
		print("average passenger wait time: ", wait_time_avg)

	x = np.linspace(0, 1, 100)[1:]
	mta_mixing_time = lambda var: np.log(5728/var)/0.0063
	mbta_mixing_time = lambda var: np.log(456/var)/0.0012
	y = np.array([mbta_mixing_time(pt)/mta_mixing_time(pt) for pt in x])
	plt.plot(x, y)
	plt.xlabel("Epsilon")
	plt.ylabel("Ratio of MBTA to MTA mixing time")
	plt.savefig('mixing_time_ratios.eps', format='eps')

spectra(mbta_file)
print("--------------------")
spectra(mta_file)