import pickle as pkl
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import os

###### the dataset ########
raw_data_path = './data/pickup_jl.csv'
city_name = 'Jilin'
abb = 'JL'
saved_dir = './data/sensor_graph/'
file_name = 'adj_mx_delivery_{}.pkl'.format(abb.lower())

#### parameters for weighted adj ####
if abb == 'SH':
    normalized_k =  0.3 #threshold
    s = 5 # scale
elif abb == 'HZ':
    normalized_k =  0.3 #threshold
    s = 5 # scale
elif abb == 'CQ':
    normalized_k =  0.4 #threshold
    s = 10 # scale
else:
    normalized_k = 0.3  # threshold
    s = 5  # scale
#####################################

# read data
df = pd.read_csv(raw_data_path)
# chose city
df_city = df[df.city == city_name]
# a list contain dipan's id
dipan_ids = list(df_city.region_id.unique())

locations = []
locations_dict = {ids:[] for ids in dipan_ids}
for dipan in dipan_ids:
    lng = np.mean(df_city[df_city.region_id == dipan].lng)
    lat = np.mean(df_city[df_city.region_id == dipan].lat)
    locations.append([lat,lng])
    locations_dict[dipan].append([lat,lng])

###Saving the location information###
data = []
for area, coords in locations_dict.items():
    for coord in coords:
        data.append([area, coord[0], coord[1]])
df = pd.DataFrame(data, columns=['regoin', 'lat', 'lng'])
df.to_csv('data/locations_jl.csv', index=False, encoding='utf-8')

###Saving the location information###
n_sensors = len(dipan_ids)
dist = np.full((n_sensors, n_sensors), np.inf)
for i in range(n_sensors):
    for j in range(n_sensors):
        dist[i][j] = geodesic(locations[i], locations[j]).kilometers

# Calculates the standard deviation as theta.
distances = dist[~np.isinf(dist)].flatten()
std = distances.std() / s

print(distances.mean(), std, distances.max(), distances.min())

adj_mx = np.exp(-np.square(dist / std))

adj_mx[adj_mx < normalized_k] = 0 # make it sparse

if not os.path.exists(saved_dir):os.mkdir(saved_dir)
with open(os.path.join(saved_dir , file_name),'wb') as f:
    pkl.dump([None,None,adj_mx],f)
print('ADJ saved in {}'.format(os.path.join(saved_dir , file_name)))