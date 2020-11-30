import requests
from urllib.parse import quote
import numpy as np
import pandas as pd
import json
import time

import itertools
from scipy.optimize import lsq_linear

def generate_route_json():
	s = requests.Session()
	s.headers.update({'accept': 'application/vnd.api+json'})

	params = {'api_key': '9ab7e3cd789443379fc2ba97f6aabc9b'}

	subway_routes = requests.get("https://api-v3.mbta.com/routes?fields%5Broute%5D=long_name%2Cfare_class&include=route_patterns", 
								params=params)
	if subway_routes.status_code != 200:
		quit()

	subway_routes = subway_routes.json()['data']

	temp = []
	for route in subway_routes:
		if route['attributes']['fare_class'] == 'Rapid Transit' and "South Station" not in route['attributes']['long_name']:
			temp.append(route)
	subway_routes = temp

	rep_trip_ids = []
	for subway_line in subway_routes:
		for route_pattern in subway_line['relationships']['route_patterns']['data']:
			rep_trip_id = requests.get("https://api-v3.mbta.com/route_patterns/%s?include=representative_trip" % quote(route_pattern['id']),
									   params=params)
			rep_trip_id = rep_trip_id.json()
			rep_trip_id = rep_trip_id['data']['relationships']['representative_trip']['data']['id']
			rep_trip_ids.append(rep_trip_id)

	trip_stop_dict = {}
	for rep_trip_id in rep_trip_ids:
		# for each element, get each route pattern and id
		get_trip_stops_str = "https://api-v3.mbta.com/trips/%s?fields%%5Btrip%%5D=stops%%2Cparent_station&include=stops" % quote(rep_trip_id)
		trip_stops = requests.get(get_trip_stops_str, params=params).json()['included']
		
		descriptive_trip_stops = []
		for stop in trip_stops:
			#stop_id = stop['id']
			#get_stop_name_str = "https://api-v3.mbta.com/stops/%s" % quote(stop_id)
			#stop_info = requests.get(get_stop_name_str, params=params).json()
			# if parent_station is present, get the id and then do stop thing again
			descriptive_trip_stops.append((stop['relationships']['parent_station']['data']['id'],
										  stop['attributes']['name']))
		trip_stop_dict[str(rep_trip_id)] = descriptive_trip_stops

	with open("mbta_trips", "w") as f:
		json.dump(trip_stop_dict, f)

def dedup_trips():
	with open("mbta_trips", "r") as f:
		trip_stop_dict = json.load(f)
		deduped_trip_stop_dict = {}
		for i in range(len(trip_stop_dict)):
			trip_id = list(trip_stop_dict.keys())[i]
			if len(deduped_trip_stop_dict) == 0:
				deduped_trip_stop_dict[(trip_id)] = trip_stop_dict[trip_id]
				continue
			was_duplicate = False
			for j in range(len(deduped_trip_stop_dict)):
				existing_trip_ids = list(deduped_trip_stop_dict.keys())[j]
				if deduped_trip_stop_dict[existing_trip_ids] == trip_stop_dict[trip_id]:
					deduped_trip_stop_dict[(existing_trip_ids, trip_id)] = trip_stop_dict[trip_id]
					deduped_trip_stop_dict.pop(existing_trip_ids)
					was_duplicate = True
					break
			if not was_duplicate:
				deduped_trip_stop_dict[(trip_id)] = trip_stop_dict[trip_id]
		
		with open("mbta_trips_deduped", "w") as f2:
			json.dump({str(k) : v for k, v in deduped_trip_stop_dict.items()}, f2)


all_places = []
trip_stop_dict = None
with open("mbta_trips_deduped", "r") as f:
	trip_stop_dict = json.load(f)
	for k, v in trip_stop_dict.items():
		for place in v:
			all_places.append(place[0])
	all_places = np.unique(np.array(all_places))
	# Verified by hand that this gets all 118 subway stations :sweat_smile:

adjacency_mat = np.zeros((len(all_places), len(all_places))).astype('int')
df = pd.DataFrame(adjacency_mat, columns=all_places, index=all_places)

for route_ids, route in trip_stop_dict.items():
	for i in range(len(route)):
		place_id = route[i][0]
		if i != len(route) - 1:
			next_place_id = route[i + 1][0]
			df[place_id][next_place_id] = 1
			df[next_place_id][place_id] = 1
		if i != 0:
			prev_place_id = route[i - 1][0]
			df[place_id][prev_place_id] = 1
			df[prev_place_id][place_id] = 1
print(df)

with open("mbta_adj_mat", "w") as f:
	df.to_json(f)


with open("mbta_adj_mat_readable", "r") as f: # might be unnecessary
    mat = pd.read_json(f).to_numpy()
    k, v = np.linalg.eig(mat)
    np.set_printoptions(suppress=True)
    print(np.all(mat == mat.T))
    D = np.diag(np.sum(mat,axis=1))
    W = np.matmul(mat, np.linalg.inv(D))
    k2, v2 = np.linalg.eig(W)
    print(k2)


with open("mbta_adj_mat_readable", "r") as f:
    M = pd.read_json(f).to_numpy()
    k, v = np.linalg.eig(M)
    np.set_printoptions(suppress=True)
    D = np.diag(np.sum(M,axis=1))
    W = np.matmul(M, np.linalg.inv(D))
    L = D-M
    omega, w = np.linalg.eig(W)
    w = np.real(w)
    omega = np.sort(np.real(omega))
    omega_pi = max(omega[-2], omega[0])
    print("expansion", 1-omega_pi)
    #quit()
    k2, v2 = np.linalg.eig(L)
    v2 = v2[:, np.argsort(k2)]
    k2 = np.sort(k2)
    print("eigenvector for 1")
    print(k2[37])
    print(v2[:, 37])
    print(np.matmul(L, v2[:, 37]) - k2[37]*v2[:, 37])
    #input()
    print("eigenvector for 2")
    print(k2[58])
    print(v2[np.argsort(k2)][58])
    print(np.matmul(L, v2[:, 58]) - k2[58]*v2[:, 58])


N = 300 # assumed total number of trains in the system
n = M.shape[0]
pi = [1/n]*n + np.random.random(n)/N
pi = pi/sum(pi) # made-up target stationary distribution

Mp = M # modified adjacency so that the Markov chain is ergodic
d = np.sum(Mp,axis=1)
print('degree:', d)
print('\n #nonzero entries:', np.sum(d))

boundary = np.argwhere(d==1).flatten()
for b in boundary:
    Mp[b,b] = 1
d = np.sum(Mp,axis=1)
print('\n modified degrees', d)

w_supp = np.argwhere(Mp!=0)
n_w = len(w_supp)

T = [[i,j] for i,j in itertools.product(range(n),range(n)) if Mp[i,j] != 0] # the coordinates we care about (~ 300 out of ~ 140k)

TI = [] # neighbor list
q = 0
for i in range(n):
    TI.append([T[k][1] for k in range(q,q+d[i])])
    q += d[i]

print(TI)

TI_lengths = [0]
q = 0
for t in TI:
    q += len(t)
    TI_lengths.append(q)

B_pi = np.zeros((n,n_w))
for i in range(n):
    B_pi[i,TI_lengths[i]:TI_lengths[i+1]] = [pi[j] for j in TI[i]]

B_1 = np.zeros((n,n_w))
for j in range(n):
    I = np.array(TI[j])
    for i in I:
        B_1[j,TI_lengths[i]+np.argwhere(np.array(TI[i])==j)[0,0]] = 1

zeta = 1
B = np.vstack([B_pi,zeta*B_1])
c = np.hstack([pi,zeta*np.ones(n)])
LB = (1/N)*np.ones(n_w)

x2 = lsq_linear(B, c, bounds=(LB, np.inf), # solution
                method='trf', tol=1e-10, lsq_solver=None, lsmr_tol=None, 
                max_iter=None, verbose=0).x

print((np.matmul(B,x2) - c)/c)

def weight(i,j,x,TI,TI_lengths):
    t = np.argwhere(np.array(TI[i]) == j).flatten()
    if len(t):
        return x[TI_lengths[i]+t[0]]
    return 0

print('e.g.', weight(0,116,x2,TI,TI_lengths))

for i in range(n):
    print(sum(x2[TI_lengths[i]:TI_lengths[i+1]]))

X = np.zeros((n,n))
for i,j in itertools.product(range(n),range(n)):
    X[i,j] = weight(i,j,x2,TI,TI_lengths)

p0 = np.array([1]+[0]*(n-1)).reshape(n,)
p = p0
t = 20000
for i in range(t):
    p = np.matmul(X,p)
print(p)
print(np.linalg.norm(p-pi,1))
			