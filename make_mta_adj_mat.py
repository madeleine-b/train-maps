import numpy as np
import pandas as pd
import json

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
	mapping = pd.read_csv("mta-data/stops.csv")
	parent_stops = mapping.loc[mapping["location_type"] == 1]
	if use_readable_names:
		return parent_stops["stop_name"].to_numpy()
	else:
		return parent_stops["stop_id"].to_numpy()

def generate_adjacency_matrix(use_readable_names=False, write_to_file=True, filename=None):
	all_places = get_all_stations(use_readable_names=use_readable_names)
	adjacency_mat = pd.DataFrame(0, columns=all_places, index=all_places)

	trip_data = pd.read_csv("mta-data/stop_times.csv")

	all_trip_ids = trip_data["trip_id"].unique()
	deduped_trip_ids = []
	for t_id in all_trip_ids:
		if t_id[:-4]+"S"+t_id[-3:] not in deduped_trip_ids and t_id[:-4]+"N"+t_id[-3:] not in deduped_trip_ids:
			deduped_trip_ids.append(t_id)
	all_trip_ids = deduped_trip_ids

	index = 1
	for trip_id in all_trip_ids:
		stops = trip_data.loc[trip_data["trip_id"] == trip_id][["stop_id", "stop_sequence"]]
		stops = stops.to_numpy()
		for i in range(len(stops)):
			place_id = stop_id_to_parent_station(stops[i][0], use_readable_names)
			if i != len(stops) - 1:
				next_place_id = stop_id_to_parent_station(stops[i + 1][0], use_readable_names)
				adjacency_mat[place_id][next_place_id] = 1
				adjacency_mat[next_place_id][place_id] = 1
			if i != 0:
				prev_place_id = stop_id_to_parent_station(stops[i - 1][0], use_readable_names)
				adjacency_mat[place_id][prev_place_id] = 1
				adjacency_mat[prev_place_id][place_id] = 1
		print("done with trip", index, "/", len(all_trip_ids))
		index += 1
	
	if write_to_file:
		if filename is None:
			filename = "mta-data/mta_adjacency_mat"
			if use_readable_station_names:
				filename += "_readable"
		with open(filename, "w") as f:
			adjacency_mat.to_json(f)
	return adjacency_mat