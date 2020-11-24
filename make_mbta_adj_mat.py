import requests
from urllib.parse import quote
import numpy
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

def remap_keys(mapping):
     return {str(k) : v for k, v in mapping.items()}

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
		json.dump(remap_keys(deduped_trip_stop_dict), f2)
			
		

		