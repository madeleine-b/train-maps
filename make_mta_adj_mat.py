import numpy as np
import pandas as pd
import json
import networkx as nx 
import matplotlib.pyplot as plt

from collections import Counter

# Inconsistencies between data files
banned_stations = {"140": "South Ferry Loop", "H19": "Broad Channel", "N12": "S.B. Coney Island"}

def stop_id_to_parent_station(stop_id, use_readable_names=False):
	mapping = pd.read_csv("mta-data/stops.csv")
	stop_row = mapping.loc[mapping["stop_id"] == stop_id]
	assert len(stop_row) == 1, "Found more than one row with stop id %s" % stop_id
	stop_row = stop_row.squeeze()
	if use_readable_names:
		return stop_row["stop_name"]
	if stop_row["location_type"] == 1:
		return stop_id
	return stop_row["parent_station"]

def get_all_stations(use_readable_names=False):
	mapping = pd.read_csv("mta-data/MTA_Stations.csv")
	mapping = mapping.loc[mapping["Line"] != "Staten Island"]
	if use_readable_names:
		return mapping["Stop_Name"].unique()
	else:
		return mapping["GTFS_Stop_ID"].unique()

def get_line_from_stop_id(stop_id):
	if stop_id in banned_stations.keys():
		return
	mapping = pd.read_csv("mta-data/MTA_Stations.csv")
	return mapping.loc[mapping["GTFS_Stop_ID"] == stop_id]["Line"].tolist()[0]
	
def generate_adjacency_matrix(use_readable_names=False, write_to_file=True, filename=None):
	all_places = get_all_stations(use_readable_names=use_readable_names)
	adjacency_mat = pd.DataFrame(0, columns=all_places, index=all_places)

	line_data = pd.read_csv("mta-data/MTA_Stations.csv")
	all_lines = line_data["Line"].unique().tolist()

	for line in all_lines:
		if line == "Staten Island":
			continue
		stops = line_data.loc[line_data["Line"] == line][["GTFS_Stop_ID", "Stop_Name"]]
		stops = stops.to_numpy()
		stop_id_index = 1 if use_readable_names else 0
		for i in range(len(stops)):
			place_id = stops[i][stop_id_index]
			if i != len(stops) - 1:
				next_place_id = stops[i + 1][stop_id_index]
				adjacency_mat[place_id][next_place_id] = 1
				adjacency_mat[next_place_id][place_id] = 1
			if i != 0:
				prev_place_id = stops[i - 1][stop_id_index]
				adjacency_mat[place_id][prev_place_id] = 1
				adjacency_mat[prev_place_id][place_id] = 1
	
	if write_to_file:
		if filename is None:
			filename = "mta-data/mta_adjacency_mat"
			if use_readable_names:
				filename += "_readable"
		with open(filename, "w") as f:
			adjacency_mat.to_json(f)
	return adjacency_mat

generate_adjacency_matrix(use_readable_names=False)

old_adj_mat = None
adj_mat = None
with open("mta-data/mta_adjacency_mat", "r") as f:
	adj_mat = pd.read_json(f)

with open("mta-data/mta_adjacency_mat_old", "r") as f:
	old_adj_mat = pd.read_json(f)

s=list(banned_stations.keys())
for b in old_adj_mat.columns:
	if get_line_from_stop_id(b) == "Staten Island":
		s.append(b)

old_adj_mat = old_adj_mat.drop(s, axis=0)
old_adj_mat = old_adj_mat.drop(s, axis=1)
for label in list(banned_stations.keys()):
	if label in adj_mat.columns:
		adj_mat = adj_mat.drop([label], axis=0)
		adj_mat = adj_mat.drop([label], axis=1)

adj_mat = adj_mat.reindex(columns=list(old_adj_mat.columns))
adj_mat = adj_mat.reindex(list(old_adj_mat.columns))

# Get edges in old adj mat that aren't in new one
edges_in_old_only = old_adj_mat - adj_mat
cols = edges_in_old_only.columns
bt = edges_in_old_only.apply(lambda x: x > 0)
bt = bt.apply(lambda x: list(cols[x.values]), axis=1)
bt = bt.loc[bt.astype(str).drop_duplicates().index]

for item, edges in bt.iteritems():
	for edge in edges:
		adj_mat[item][edge] = 1
		adj_mat[edge][item] = 1

# Combine stations that have the same "human-readable" name
full_name_labels = [(s, stop_id_to_parent_station(s,True)) for s in adj_mat.columns]
counts = Counter([stop_id_to_parent_station(s,True) for s in adj_mat.columns])
columns_to_combine = []
for s in full_name_labels:
	if counts[s[1]] > 1:
		dups = [l[0] for l in full_name_labels if l[1] == s[1]]
		columns_to_combine.append(dups)

removed_stop_ids = []
for dups in columns_to_combine:
	if set(dups[1:]) in removed_stop_ids:
		continue
	adjacency_sum = adj_mat[dups].sum(axis=1)
	adj_mat = adj_mat.drop(dups, axis=0)
	adj_mat = adj_mat.drop(dups, axis=1)
	adj_mat[dups[0]] = adjacency_sum
	adj_mat.loc[dups[0]] = adjacency_sum
	removed_stop_ids.append(set(dups[1:]))

G = nx.from_numpy_matrix(np.array(adj_mat)) 
label_dict = {}
for stop_index in range(len(adj_mat.columns)):
	stop = adj_mat.columns[stop_index]
	label_dict[stop_index] = stop_id_to_parent_station(stop, True)
nx.draw(G, labels=label_dict, font_size=8) 
plt.show()