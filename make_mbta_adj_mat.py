import requests
from urllib.parse import quote
import numpy as np
import pandas as pd
import json
import time

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

			