import requests
from urllib.parse import quote
import numpy as np
import pandas as pd
import json
import time
import random
import itertools
from scipy.optimize import lsq_linear
from math import floor, ceil

import networkx as nx 
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)


# Hand-counted the rolling stock 
TOTAL_MBTA_TRAINS = 171

MBTA_SUBWAY_STATIONS = set("Alewife, Andrew, Ashmont, Braintree, Broadway, Central, Charles/MGH, Davis,"\
                       " Downtown Crossing, Fields Corner, Harvard, JFK/UMass, Kendall/MIT, North Quincy,"\
                       " Park Street, Porter, Quincy Adams, Quincy Center, Savin Hill, Shawmut, South Station,"\
                       " Wollaston, Assembly, Back Bay, Chinatown, Community College, Downtown Crossing,"\
                       " Forest Hills, Green Street, Haymarket, Jackson Square, Malden Center,"\
                       " Massachusetts Avenue, North Station, Oak Grove, Roxbury Crossing, Ruggles, State,"\
                       " Stony Brook, Sullivan Square, Tufts Medical Center, Wellington, Allston Street,"\
                       " Arlington, Babcock Street, Back of the Hill, Beaconsfield, Blandford Street,"\
                       " Boston College, Boston University Central, Boston University East, Boston University West,"\
                       " Boylston, Brandon Hall, Brigham Circle, Brookline Hills, Brookline Village, Chestnut Hill,"\
                       " Chestnut Hill Avenue, Chiswick Road, Cleveland Circle, Coolidge Corner, Copley, Dean Road,"\
                       " Eliot, Englewood Avenue, Fairbanks Street, Fenway, Fenwood Road, Government Center,"\
                       " Griggs Street, Harvard Avenue, Hawes Street, Haymarket, Heath Street,"\
                       " Hynes Convention Center, Kenmore, Kent Street, Lechmere, Longwood, Longwood Medical Area,"\
                       " Mission Park, Museum of Fine Arts, Newton Centre, Newton Highlands, North Station,"\
                       " Northeastern University, Packards Corner, Park Street, Pleasant Street, Prudential,"\
                       " Reservoir, Riverside, Riverway, Saint Mary's Street, Saint Paul Street (B), Saint Paul Street (C),"\
                       " South Street, Summit Avenue, Sutherland Road, Symphony, Tappan Street, Waban, Warren Street,"\
                       " Washington Square, Washington Street, Woodland, Airport, Aquarium, Beachmont, Bowdoin,"\
                       " Government Center, Maverick, Orient Heights, Revere Beach, State, Science Park/West End,"\
                       "Suffolk Downs, Wonderland, Wood Island".replace(", ", ",").split(","))

MBTA_SUBWAY_IDS = ['place-alfcl', 'place-alsgr', 'place-andrw', 'place-aport', 'place-aqucl', 'place-armnl', 
                   'place-asmnl', 'place-astao', 'place-babck', 'place-bbsta', 'place-bckhl', 'place-bcnfd', 
                   'place-bcnwa', 'place-bland', 'place-bmmnl', 'place-bndhl', 'place-bomnl', 'place-boyls', 
                   'place-brdwy', 'place-brico', 'place-brkhl', 'place-brmnl', 'place-brntn', 'place-bucen', 
                   'place-buest', 'place-buwst', 'place-bvmnl', 'place-ccmnl', 'place-chhil', 'place-chill', 
                   'place-chmnl', 'place-chncl', 'place-chswk', 'place-clmnl', 'place-cntsq', 'place-coecl', 
                   'place-cool', 'place-davis', 'place-denrd', 'place-dwnxg', 'place-eliot', 'place-engav', 
                   'place-fbkst', 'place-fenwd', 'place-fenwy', 'place-fldcr', 'place-forhl', 'place-gover', 
                   'place-grigg', 'place-grnst', 'place-haecl', 'place-harsq', 'place-harvd', 'place-hsmnl', 
                   'place-hwsst', 'place-hymnl', 'place-jaksn', 'place-jfk', 'place-kencl', 'place-knncl', 
                   'place-kntst', 'place-lake', 'place-lngmd', 'place-longw', 'place-masta', 'place-mfa', 
                   'place-mispk', 'place-mlmnl', 'place-mvbcl', 'place-newtn', 'place-newto', 'place-north', 
                   'place-nqncy', 'place-nuniv', 'place-ogmnl', 'place-orhte', 'place-pktrm', 'place-plsgr', 
                   'place-portr', 'place-prmnl', 'place-qamnl', 'place-qnctr', 'place-rbmnl', 'place-rcmnl', 
                   'place-river', 'place-rsmnl', 'place-rugg', 'place-rvrwy', 'place-sbmnl', 'place-sdmnl', 
                   'place-shmnl', 'place-smary', 'place-smmnl', 'place-sougr', 'place-sstat', 'place-state', 
                   'place-sthld', 'place-stplb', 'place-stpul', 'place-sull', 'place-sumav', 'place-symcl', 
                   'place-tapst', 'place-tumnl', 'place-waban', 'place-wascm', 'place-welln', 'place-wimnl', 
                   'place-wlsta', 'place-wondl', 'place-woodl', 'place-wrnst', 'place-lech', 'place-spmnl']

assert len(MBTA_SUBWAY_STATIONS) == len(MBTA_SUBWAY_IDS), "Something in our station ground truth got misaligned!"

def stop_id_to_name(stop_id):
    mapping = {'place-alsgr': 'Allston Street', 'place-armnl': 'Arlington', 'place-babck': 'Babcock Street', 
               'place-bckhl': 'Back of the Hill', 'place-bcnfd': 'Beaconsfield', 'place-bcnwa': 'Washington Square', 
               'place-bland': 'Blandford Street', 'place-bndhl': 'Brandon Hall', 'place-boyls': 'Boylston', 
               'place-brico': 'Packards Corner', 'place-brkhl': 'Brookline Hills', 'place-brmnl': 'Brigham Circle', 
               'place-bucen': 'Boston University Central', 'place-buest': 'Boston University East', 
               'place-buwst': 'Boston University West', 'place-bvmnl': 'Brookline Village', 'place-chhil': 'Chestnut Hill', 
               'place-chill': 'Chestnut Hill Avenue', 'place-chswk': 'Chiswick Road', 'place-clmnl': 'Cleveland Circle', 
               'place-coecl': 'Copley', 'place-cool': 'Coolidge Corner', 'place-denrd': 'Dean Road', 'place-eliot': 'Eliot', 
               'place-engav': 'Englewood Avenue', 'place-fbkst': 'Fairbanks Street', 'place-fenwd': 'Fenwood Road', 
               'place-fenwy': 'Fenway', 'place-gover': 'Government Center', 'place-grigg': 'Griggs Street',
               'place-haecl': 'Haymarket', 'place-harvd': 'Harvard Avenue', 'place-hsmnl': 'Heath Street', 
               'place-hwsst': 'Hawes Street', 'place-hymnl': 'Hynes Convention Center', 'place-kencl': 'Kenmore', 
               'place-kntst': 'Kent Street', 'place-lake': 'Boston College', 'place-lech': 'Lechmere', 
               'place-lngmd': 'Longwood Medical Area', 'place-longw': 'Longwood', 'place-mfa': 'Museum of Fine Arts',
               'place-mispk': 'Mission Park', 'place-newtn': 'Newton Highlands', 'place-newto': 'Newton Centre', 
               'place-north': 'North Station', 'place-nuniv': 'Northeastern University', 'place-pktrm': 'Park Street', 
               'place-plsgr': 'Pleasant Street', 'place-prmnl': 'Prudential', 'place-river': 'Riverside', 
               'place-rsmnl': 'Reservoir', 'place-rvrwy': 'Riverway', 'place-smary': "Saint Mary's Street", 
               'place-sougr': 'South Street', 'place-spmnl': 'Science Park/West End', 'place-sthld': 'Sutherland Road', 
               'place-stplb': 'Saint Paul Street (B)', 'place-stpul': 'Saint Paul Street (C)', 'place-sumav': 'Summit Avenue', 
               'place-symcl': 'Symphony', 'place-tapst': 'Tappan Street', 'place-waban': 'Waban', 
               'place-wascm': 'Washington Street', 'place-woodl': 'Woodland', 'place-wrnst': 'Warren Street', 
               'place-bmmnl': 'Beachmont', 'place-bomnl': 'Bowdoin', 'place-mvbcl': 'Maverick', 
               'place-orhte': 'Orient Heights', 'place-rbmnl': 'Revere Beach', 'place-sdmnl': 'Suffolk Downs', 
               'place-state': 'State', 'place-wimnl': 'Wood Island', 'place-wondl': 'Wonderland', 
               'place-aport': 'Airport', 'place-aqucl': 'Aquarium', 'place-astao': 'Assembly', 
               'place-bbsta': 'Back Bay', 'place-ccmnl': 'Community College', 'place-chncl': 'Chinatown', 
               'place-dwnxg': 'Downtown Crossing', 'place-forhl': 'Forest Hills', 'place-grnst': 'Green Street', 
               'place-jaksn': 'Jackson Square', 'place-masta': 'Massachusetts Avenue', 'place-mlmnl': 'Malden Center', 
               'place-ogmnl': 'Oak Grove', 'place-rcmnl': 'Roxbury Crossing', 'place-rugg': 'Ruggles', 
               'place-sbmnl': 'Stony Brook', 'place-sull': 'Sullivan Square', 'place-tumnl': 'Tufts Medical Center', 
               'place-welln': 'Wellington', 'place-alfcl': 'Alewife', 'place-andrw': 'Andrew', 'place-asmnl': 'Ashmont', 
               'place-brdwy': 'Broadway', 'place-brntn': 'Braintree', 'place-chmnl': 'Charles/MGH', 
               'place-cntsq': 'Central', 'place-davis': 'Davis', 'place-fldcr': 'Fields Corner', 
               'place-harsq': 'Harvard', 'place-jfk': 'JFK/UMass', 'place-knncl': 'Kendall/MIT', 
               'place-nqncy': 'North Quincy', 'place-portr': 'Porter', 'place-qamnl': 'Quincy Adams', 
               'place-qnctr': 'Quincy Center', 'place-shmnl': 'Savin Hill', 'place-smmnl': 'Shawmut', 
               'place-sstat': 'South Station', 'place-wlsta': 'Wollaston'}
    return mapping[stop_id]

def stop_name_to_id(stop_name):
  for s_id in MBTA_SUBWAY_IDS:
    if stop_id_to_name(s_id) == stop_name:
      return s_id
  assert False, "Stop name %s not found" % stop_name

def generate_route_json():
    confirm = input("Are you sure you want to grab current MBTA schedule information? This may mess up things Y/N")
    if confirm != "Y":
      return
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
            rep_trip_info = rep_trip_id.json()
            rep_trip_id = rep_trip_info['data']['relationships']['representative_trip']['data']['id']
            rep_trip_ids.append(rep_trip_id)

    trip_stop_dict = {}
    for rep_trip_id in rep_trip_ids:
        # for each element, get each route pattern and id
        get_trip_stops_str = "https://api-v3.mbta.com/trips/%s?fields%%5Btrip%%5D=stops%%2Cparent_station&include=stops" % quote(rep_trip_id)
        trip_stops = requests.get(get_trip_stops_str, params=params).json()

        trip_stops = trip_stops['included']
        
        descriptive_trip_stops = []
        for stop in trip_stops:
            stop_id = stop['relationships']['parent_station']['data']['id']
            stop_name = stop['attributes']['name']
            if stop_id == "place-stplb":
                stop_name = "Saint Paul Street (B)"
            elif stop_id == "place-stpul":
                stop_name = "Saint Paul Street (C)"
            # if parent_station is present, get the id and then do stop thing again
            descriptive_trip_stops.append((stop_id, stop_name))
        trip_stop_dict[str(rep_trip_id)] = descriptive_trip_stops

    with open("mbta-data/mbta_trips", "w") as f:
        json.dump(trip_stop_dict, f)

def dedup_trips():
    with open("mbta-data/mbta_trips", "r") as f:
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
        
        with open("mbta-data/mbta_trips_deduped", "w") as f2:
            json.dump({str(k) : v for k, v in deduped_trip_stop_dict.items()}, f2)

def adjust_mbta_hardcoded_stations(adj_mat, use_readable_station_names):
    hardcoded_connections = {"place-boyls": ["place-armnl", "place-pktrm"], 
                             "place-lech" : ["place-spmnl"],
                             "place-spmnl": ["place-lech", "place-north"],
                             "place-bland": ["place-buest", "place-kencl"],
                             "place-bomnl": ["place-gover"],
                             "place-hymnl": ["place-kencl", "place-coecl"],
                             "place-kencl": ["place-smary", "place-bland", "place-fenwy", "place-hymnl"],
                             "place-north": ["place-spmnl", "place-ccmnl", "place-haecl"], 
                             "place-orhte": ["place-sdmnl", "place-wimnl"],
                             "place-pktrm": ["place-chmnl", "place-dwnxg", "place-boyls", "place-gover"],
                             "place-welln": ["place-astao", "place-mlmnl"],
                             "place-astao": ["place-welln", "place-sull"],
                             "place-prmnl": ["place-coecl", "place-symcl"]}
    
    for stop_id, connections in hardcoded_connections.items():
        stop_name = stop_id_to_name(stop_id) if use_readable_station_names else stop_id
        if stop_name not in adj_mat.columns:
            adj_mat.insert(len(adj_mat.columns), stop_name, np.zeros(len(adj_mat), dtype=np.int64))
        adj_mat.loc[stop_name, :] = 0
        adj_mat.loc[:, stop_name] = 0

    for stop_id, connections in hardcoded_connections.items():
        stop_name = stop_id_to_name(stop_id) if use_readable_station_names else stop_id
        for neighbor_id in connections:
            neighbor_name = stop_id_to_name(neighbor_id) if use_readable_station_names else neighbor_id
            adj_mat[stop_name][neighbor_name] = 1
            adj_mat[neighbor_name][stop_name] = 1


def generate_adj_matrix(use_readable_station_names=False, write_to_file=True, filename=None):
    stop_info_index = 1 if use_readable_station_names else 0
    all_places = []
    trip_stop_dict = None
    # IGNORE MATTAPAN TROLLEY
    if not use_readable_station_names:
        mattapan_stations = ['place-butlr', 'place-capst', 'place-cedgr', 'place-cenav', 
                             'place-matt', 'place-miltt', 'place-valrd']
        
    else:
        mattapan_stations = ["Butler", "Capen Street", "Cedar Grove", "Central Avenue", 
                             "Mattapan", "Milton", "Valley Road"]
        
    with open("mbta-data/mbta_trips_deduped", "r") as f:
        trip_stop_dict = json.load(f)
        for k, v in trip_stop_dict.items():
            for place in v:
                if place[stop_info_index]:
                    all_places.append(place[stop_info_index])
        all_places = np.unique(np.array(all_places))

    df = pd.DataFrame(0, columns=all_places, index=all_places)

    for route_ids, route in trip_stop_dict.items():
        for i in range(len(route)):
            place_id = route[i][stop_info_index]
            if i != len(route) - 1:
                next_place_id = route[i + 1][stop_info_index]
                df[place_id][next_place_id] = 1
                df[next_place_id][place_id] = 1
            if i != 0:
                prev_place_id = route[i - 1][stop_info_index]
                df[place_id][prev_place_id] = 1
                df[prev_place_id][place_id] = 1

    adjust_mbta_hardcoded_stations(df, use_readable_station_names)

    df = df.drop(mattapan_stations, axis=0)
    df = df.drop(mattapan_stations, axis=1)
    assert len(df.columns) == len(MBTA_SUBWAY_STATIONS), \
            "The MBTA API gave us %d stations but we expected %d" % (len(all_places), len(MBTA_SUBWAY_STATIONS))

    if write_to_file:
        if filename is None:
            filename = "mbta_adj_mat"
            if use_readable_station_names:
                filename += "_readable"
        with open(filename, "w") as f:
            df.to_json(f)
    return df


def create_pi(data, N):
    def func(row):
        global total_riders
        total_riders += row["total_ons"]
        return (row["stop_id"], row["total_ons"])
    pi = {}
    stop_stats = data.apply(func, axis=1)
    for stop in stop_stats:
        if stop[0] not in pi:
            pi[stop[0]] = 0
        pi[stop[0]] += stop[1]
    for stop in pi.keys():
        pi[stop] /= total_riders
        pi[stop] = max(pi[stop], 1/N)
    # Rescale to have norm 1 since we did `max` above
    total_sum = sum([pi[stop] for stop in pi.keys()])
    for stop in pi.keys():
        pi[stop] /= total_sum
    return pi

total_riders = 0

# line and stop data from https://mbta-massdot.opendata.arcgis.com/datasets/mbta-rail-ridership-by-time-period-season-route-line-and-stop
# Let's pick Fall 2019 PM_PEAK data
line_data = pd.read_csv("mbta-data/MBTA data.csv")
fall_2019_line_data = line_data.loc[(line_data["season"] == "Fall 2019") & (line_data["time_period_name"] == "PM_PEAK")]

pi = create_pi(fall_2019_line_data, TOTAL_MBTA_TRAINS)

adj_mat_df = generate_adj_matrix(write_to_file=False)

adj_mat_df_cols = list(adj_mat_df.columns)
# Reorder so the corresponding elements align with the adjacency matrix
pi = np.array([pi[stop] for stop in adj_mat_df_cols])
assert len(pi) == len(adj_mat_df_cols), "Mismatch in shape of pi and M"

adj_mat = adj_mat_df.to_numpy().astype('int')

orange_stations = ["Forest Hills","Green Street","Stony Brook","Jackson Square","Roxbury Crossing",
                   "Ruggles","Massachusetts Avenue","Back Bay","Tufts Medical Center","Chinatown",
                   "Downtown Crossing","State","Haymarket","North Station","Community College",
                   "Sullivan Square","Assembly","Wellington","Malden Center","Oak Grove"]
blue_stations = ["Bowdoin","Government Center","State","Aquarium","Maverick","Airport","Wood Island",
                 "Orient Heights","Suffolk Downs","Beachmont","Revere Beach",
                 "Wonderland"]
red_common = ["Alewife","Davis","Porter","Harvard","Central","Kendall/MIT","Charles/MGH",
              "Park Street","Downtown Crossing","South Station","Broadway","Andrew","JFK/UMass"]
red_ashmont = ["Savin Hill","Fields Corner","Shawmut","Ashmont"]
red_braintree = ["North Quincy","Wollaston","Quincy Center","Quincy Adams","Braintree"]

#green_common = ["Lechmere","Science Park/West End","Haymarket","Government Center","Park Street","Boylston","Arlington","Copley"]
green_e = ["Heath Street","Back of the Hill","Riverway","Mission Park","Fenwood Road","Brigham Circle",
            "Longwood Medical Area","Museum of Fine Arts","Northeastern University","Symphony","Prudential",
            "Copley", "Arlington", "Boylston","Park Street","Government Center","Haymarket", "North Station", 
            "Science Park/West End", "Lechmere"]

green_d = ["Riverside","Woodland","Waban","Eliot","Newton Highlands","Newton Centre","Chestnut Hill",
            "Reservoir","Beaconsfield","Brookline Hills","Brookline Village","Longwood","Fenway",
            "Kenmore","Hynes Convention Center", "Copley", "Arlington", "Boylston","Park Street","Government Center"]

green_b = ["Boston College","South Street","Chestnut Hill Avenue","Chiswick Road","Sutherland Road",
            "Washington Street","Warren Street","Allston Street","Griggs Street","Harvard Avenue",
            "Packards Corner","Babcock Street","Pleasant Street","Saint Paul Street (B)",
            "Boston University West","Boston University Central","Boston University East","Blandford Street",
            "Kenmore","Hynes Convention Center", "Copley", "Arlington", "Boylston","Park Street"]

green_c = ["Cleveland Circle","Englewood Avenue","Dean Road","Tappan Street","Washington Square",
            "Fairbanks Street","Brandon Hall","Summit Avenue","Coolidge Corner","Saint Paul Street (C)",
            "Kent Street","Hawes Street","Saint Mary's Street", "Kenmore","Hynes Convention Center", 
            "Copley", "Arlington", "Boylston","Park Street", "Government Center","Haymarket", "North Station"]

# Stations which are allowed to hold more trains that outwardly appearing connections
# e.g. Government Center has 2 green line tracks plus a non-revenue loop since D-line trains
# turn around there
terminal_stations = ["Alewife", "Ashmont", "Braintree", "Forest Hills", "Oak Grove", "Wonderland", 
                     "Bowdoin", "Heath Street", "Boston College", "Riverside", "Cleveland Circle",
                      "Lechmere", "Park Street", "Government Center", "North Station"]

red_stations = list(set(red_common + red_braintree + red_ashmont))
green_stations = list(set(green_b + green_c + green_e + green_d))

real_schedule_adj_mat = {}
max_blue_trains = 15
max_green_trains = 99
max_red_trains = 34
max_orange_trains = 23
t = 0
valve_count = 0
total_trains_along_each_edge = pd.DataFrame(0, columns=MBTA_SUBWAY_STATIONS, index=MBTA_SUBWAY_STATIONS)
real_schedule_adj_mat = {"Orange" : {}, "Red" : {}, "Green" : {}, "Blue" : {}}

d = {}
for s in orange_stations:
  d[s] = max_orange_trains // len(orange_stations)
d["Forest Hills"] += floor((max_orange_trains % len(orange_stations)) / 2)
d["Oak Grove"] += ceil((max_orange_trains % len(orange_stations)) / 2)
real_schedule_adj_mat["Orange"] = d

d = {}
for s in blue_stations:
  d[s] = max_blue_trains // len(blue_stations)
d["Wonderland"] += floor((max_blue_trains % len(blue_stations)) / 2)
d["Bowdoin"] += ceil((max_blue_trains % len(blue_stations)) / 2)
real_schedule_adj_mat["Blue"] = d

d = {}
for s in green_stations:
  d[s] = max_green_trains // len(green_stations)
stations_with_second_train = random.sample(green_stations, k=(max_green_trains % len(green_stations)))
for s in stations_with_second_train:
  d[s] += 1
real_schedule_adj_mat["Green"] = d

d = {}
for s in red_stations:
  d[s] = max_red_trains // len(red_stations)
stations_with_second_train = random.sample(red_stations, k=(max_red_trains % len(red_stations)))
for s in stations_with_second_train:
  d[s] += 1
real_schedule_adj_mat["Red"] = d

def out_edges_of_stop(name):
  neighs = adj_mat_df.columns[adj_mat_df[stop_name_to_id(name)].to_numpy().nonzero()[0]]
  return [stop_id_to_name(neigh) for neigh in neighs]

# Returns True if sending a train to each node in |outs|
# and applying |update| would result in a station trying to hold more
# trains than tracks it has (which we assume to be the number of
# neighbors unless it's a terminal station)
def invalid_update(outs, update, current_adj_mat):
  for node in outs:
    if node in update and \
       node not in terminal_stations and \
      (current_adj_mat[node] + update[node] + 1) > len(out_edges_of_stop(node)):
      return True
  return False

while t < 180:
  if t % 9 == 0:
    # move red line trains
    update = {}
    for node in red_stations:
      all_outs = out_edges_of_stop(node)
      all_outs = [node for node in all_outs if node in red_stations]
      outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Red"][node], len(all_outs)))
      for i in range(len(outs)):
        update[outs[i]] = 1 if outs[i] not in update else update[outs[i]] + 1
        update[node] = -1 if node not in update else update[node] - 1
        total_trains_along_each_edge[outs[i]][node] += 1
      if real_schedule_adj_mat["Red"][node] > len(outs):
        # Kept some at station, i.e. self-loop
        # Don't put in adjacency matrix so we don't CHOOSE to keep trains we could send
        total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Red"][node] - len(outs)
        if real_schedule_adj_mat["Red"][node] > len(out_edges_of_stop(node)):
          assert node in terminal_stations, \
                "Trying to keep %d trains at %s but it isn't a terminal station" % (real_schedule_adj_mat["Red"][node], node)
    for node in update:
      real_schedule_adj_mat["Red"][node] += update[node]
      
  if t % 5 == 0:
    # move blue line trains
    update = {}
    for node in blue_stations:
      all_outs = out_edges_of_stop(node)
      all_outs = [node for node in all_outs if node in blue_stations]
      outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Blue"][node], len(all_outs)))
      for i in range(len(outs)):
        update[outs[i]] = 1 if outs[i] not in update else update[outs[i]] + 1
        update[node] = -1 if node not in update else update[node] - 1
        total_trains_along_each_edge[outs[i]][node] += 1
      if real_schedule_adj_mat["Blue"][node] > len(outs):
        # Kept some at station, i.e. self-loop
        # Don't put in adjacency matrix so we don't CHOOSE to keep trains we could send
        total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Blue"][node] - len(outs)
        if real_schedule_adj_mat["Blue"][node] > len(out_edges_of_stop(node)):
          assert node in terminal_stations, \
                "Trying to keep %d trains at %s but it isn't a terminal station" % (real_schedule_adj_mat["Blue"][node], node)
    for node in update:
      real_schedule_adj_mat["Blue"][node] += update[node]
      
  if t % 6 == 0:
    # move orange line trains
    update = {}
    for node in orange_stations:
      all_outs = out_edges_of_stop(node)
      all_outs = [node for node in all_outs if node in orange_stations]
      outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Orange"][node], len(all_outs)))
      for i in range(len(outs)):
        update[outs[i]] = 1 if outs[i] not in update else update[outs[i]] + 1
        update[node] = -1 if node not in update else update[node] - 1
        total_trains_along_each_edge[outs[i]][node] += 1
      if real_schedule_adj_mat["Orange"][node] > len(outs):
        # Kept some at station, i.e. self-loop
        # Don't put in adjacency matrix so we don't CHOOSE to keep trains we could send
        assert node in terminal_stations, "Node %s isn't a terminal station" % node
        total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Orange"][node] - len(outs)
    for node in update:
      real_schedule_adj_mat["Orange"][node] += update[node]
    
    # move D and B line trains
    if t % 7 != 0:
      update = {}
      joint_stations = list(set(green_d + green_b))
      for node in joint_stations:
        all_outs = out_edges_of_stop(node)
        all_outs = [node for node in all_outs if node in joint_stations]
        outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Green"][node], len(all_outs)))
        retries = 100
        while node not in terminal_stations and invalid_update(outs, update, real_schedule_adj_mat["Green"]):
          outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Green"][node], len(all_outs)))
          retries -= 1
          if retries < 0:
            outs = []
            break
        for i in range(len(outs)):
          update[outs[i]] = 1 if outs[i] not in update else update[outs[i]] + 1
          update[node] = -1 if node not in update else update[node] - 1
          total_trains_along_each_edge[outs[i]][node] += 1
      for node in update:
        real_schedule_adj_mat["Green"][node] += update[node]
      for node in update:
        if real_schedule_adj_mat["Green"][node] > len(out_edges_of_stop(node)):
          if node not in terminal_stations:
            # The Green line is weird. I think this is close to what the MBTA does—sometimes they just
            # send trains to the other branches. So, we'll do that if we are running into a satisfiability problem.
            random_terminal = random.choice(["Lechmere", "Boston College", "Cleveland Circle", "Riverside", "Heath Street"])
            extra_trains = real_schedule_adj_mat["Green"][node] - len(out_edges_of_stop(node))
            real_schedule_adj_mat["Green"][node] = len(out_edges_of_stop(node))
            real_schedule_adj_mat["Green"][random_terminal] += extra_trains
            valve_count += 1
          else:
            # Kept some at station, i.e. self-loop
            # We don't add the self-loop to the adjacency matrix so as to never keep trains we could send
            if node not in update or update[node] <= 0:
              # if we had a net outflow of trains, we kept our current number
              total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Green"][node] 
            else:
              # we kept our current number minus the number we received
              total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Green"][node] - update[node]
    
  if t % 7 == 0:
    # move C and E line trains
    if t % 6 != 0:
      update = {}
      joint_stations = list(set(green_c + green_e))
      for node in joint_stations:
        all_outs = out_edges_of_stop(node)
        all_outs = [node for node in all_outs if node in joint_stations]
        #print(min(real_schedule_adj_mat["Green"][node], len(all_outs)))
        outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Green"][node], len(all_outs)))
        retries = 100
        while node not in terminal_stations and invalid_update(outs, update, real_schedule_adj_mat["Green"]):
          outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Green"][node], len(all_outs)))
          retries -= 1
          if retries < 0:
            outs = []
            break
        for i in range(len(outs)):
          update[outs[i]] = 1 if outs[i] not in update else update[outs[i]] + 1
          update[node] = -1 if node not in update else update[node] - 1
          total_trains_along_each_edge[outs[i]][node] += 1
      for node in update:
        real_schedule_adj_mat["Green"][node] += update[node]
      for node in update:
        if real_schedule_adj_mat["Green"][node] > len(out_edges_of_stop(node)):
          if node not in terminal_stations:
            # The Green line is weird. I think this is close to what the MBTA does—sometimes they just
            # send trains to the other branches. So, we'll do that if we are running into a satisfiability problem.
            random_terminal = random.choice(["Lechmere", "Boston College", "Cleveland Circle", "Riverside", "Heath Street"])
            extra_trains = real_schedule_adj_mat["Green"][node] - len(out_edges_of_stop(node))
            real_schedule_adj_mat["Green"][node] = len(out_edges_of_stop(node))
            real_schedule_adj_mat["Green"][random_terminal] += extra_trains
            valve_count += 1
          else:
            # Kept some at station, i.e. self-loop
            # We don't add the self-loop to the adjacency matrix so as to never keep trains we could send
            if node not in update or update[node] <= 0:
              # if we had a net outflow of trains, we kept our current number
              total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Green"][node] 
            else:
              # we kept our current number minus the number we received
              total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Green"][node] - update[node]

  if t % 7 == 0 and t % 6 == 0:
    # simultaneously moving all green line trains
    update = {}
    for node in green_stations:
      all_outs = out_edges_of_stop(node)
      all_outs = [node for node in all_outs if node in green_stations]
      outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Green"][node], len(all_outs)))
      retries = 100
      while node not in terminal_stations and invalid_update(outs, update, real_schedule_adj_mat["Green"]):
        outs = random.sample(all_outs, k=min(real_schedule_adj_mat["Green"][node], len(all_outs)))
        retries -= 1
        if retries < 0:
          outs = []
          break
      for i in range(len(outs)):
        update[outs[i]] = 1 if outs[i] not in update else update[outs[i]] + 1
        update[node] = -1 if node not in update else update[node] - 1
        total_trains_along_each_edge[outs[i]][node] += 1
    for node in update:
        real_schedule_adj_mat["Green"][node] += update[node]
    for node in update:
        if real_schedule_adj_mat["Green"][node] > len(out_edges_of_stop(node)):
          if node not in terminal_stations:
            # The Green line is weird. I think this is close to what the MBTA does—sometimes they just
            # send trains to the other branches. So, we'll do that if we are running into a satisfiability problem.
            random_terminal = random.choice(["Lechmere", "Boston College", "Cleveland Circle", "Riverside", "Heath Street"])
            extra_trains = real_schedule_adj_mat["Green"][node] - len(out_edges_of_stop(node))
            real_schedule_adj_mat["Green"][node] = len(out_edges_of_stop(node))
            real_schedule_adj_mat["Green"][random_terminal] += extra_trains
            valve_count += 1
          else:
            # Kept some at station, i.e. self-loop
            # We don't add the self-loop to the adjacency matrix so as to never keep trains we could send
            if node not in update or update[node] <= 0:
              # if we had a net outflow of trains, we kept our current number
              total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Green"][node] 
            else:
              # we kept our current number minus the number we received
              total_trains_along_each_edge[node][node] += real_schedule_adj_mat["Green"][node] - update[node]
  t += 1

total = 0
for node in orange_stations:
  total += real_schedule_adj_mat["Orange"][node]
assert total == max_orange_trains
total = 0
for node in blue_stations:
  total += real_schedule_adj_mat["Blue"][node]
assert total == max_blue_trains
total = 0
for node in red_stations:
  total += real_schedule_adj_mat["Red"][node]
assert total == max_red_trains
total = 0
for node in green_stations:
  total += real_schedule_adj_mat["Green"][node]
assert total == max_green_trains
print("Used green line escape valve %d times" % valve_count)
#print(total_trains_along_each_edge.to_numpy() / 180)
with open("total_trains_along_each_edge.csv", "w") as f:
  total_trains_along_each_edge.to_csv(f)

quit()
def opt_setup(M,pi,N=N,zeta=1):
    n = M.shape[0]
    Mp = M.copy() # modified adjacency so that the Markov chain is ergodic
    d = np.sum(Mp,axis=1)
    #print('degrees:', d)
    #print('\n #nonzero weights:', np.sum(d))

    boundary = np.argwhere(d==1).flatten()
    for b in boundary: # make M irreducible
        Mp[b,b] = 1
    d = np.sum(Mp,axis=1)

    w_supp = np.argwhere(Mp!=0)
    n_w = len(w_supp)

    edges = [[i,j] for i,j in itertools.product(range(n),range(n)) if Mp[i,j] != 0] # the edges we care about (~ 300 out of ~ 140k)

    neighbors = [] # neighbor list: neighbors[i] are neighbors of vertex i
    q = 0
    for i in range(n):
        neighbors.append([edges[k][1] for k in range(q,q+d[i])])
        q += d[i]

    print(neighbors)

    lengths = [0]
    q = 0
    for t in neighbors:
        q += len(t)
        lengths.append(q)

    B_pi = np.zeros((n,n_w))
    for i in range(n):
        B_pi[i,lengths[i]:lengths[i+1]] = [pi[j] for j in neighbors[i]]

    B_1 = np.zeros((n,n_w))
    for j in range(n):
        I = np.array(neighbors[j])
        for i in I:
            B_1[j,lengths[i]+np.argwhere(np.array(neighbors[i])==j)[0,0]] = 1
    
    B_detail = np.zeros((n,n_w))
    for i in range(n):
        row1 = np.zeros(n_w)
        row1[lengths[i]:lengths[i+1]] = [pi[j] for j in neighbors[i]]
        row2 = np.zeros(n_w)
        K = np.array(neighbors[i])
        for k in K:
            row2[lengths[k]+np.argwhere(np.array(neighbors[k])==i)[0,0]] = pi[i]
        B_detail[i] = row1 - row2

    B = np.vstack([B_pi,zeta*B_1,B_detail])
    c = np.hstack([pi,zeta*np.ones(n),np.zeros(n)])
    LB = (1/N)*np.ones(n_w)
    
    return B,c,LB,neighbors,lengths
    
B,c,LB,neighbors,lengths = opt_setup(adj_mat,pi)

x1 = lsq_linear(B, c, bounds=(LB, np.inf), # solution
                method='trf', tol=1e-10, lsq_solver=None, lsmr_tol=None, 
                max_iter=None, verbose=0).x
                
def weight(i,j,x,neighbors,lengths):
    t = np.argwhere(np.array(neighbors[i]) == j).flatten()
    if len(t):
        return x[lengths[i]+t[0]]
    return 0
    
n = B.shape[0]//3
W_sol = np.zeros((n,n))
for i,j in itertools.product(range(n),range(n)):
    W_sol[i,j] = weight(i,j,x1,neighbors,lengths)

M_sol = np.matmul(W_sol,np.diag(1/pi))

E = np.sum(M_sol,0)/np.sum(M_sol,1) + np.sum(M_sol,1)/np.sum(M_sol,0)

print(E)

plt.hist(E,bins=100)
plt.show()

