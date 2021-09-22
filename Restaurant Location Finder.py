#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>Restaurant Location Finder</h1>

# In this scenario, it is urgent to adopt machine learning tools in order to assist homebuyers in US to make wise and effective decisions. As a result, the business problem we are currently posing is: how could we provide support to homebuyers in USA to purchase a suitable real estate in this uncertain economic and financial scenario

# In[ ]:


import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values


from IPython.display import Image 
from IPython.core.display import HTML 
    

from pandas.io.json import json_normalize

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # plotting library

print('Folium installed')
print('Libraries imported.')


# In[7]:


CLIENT_ID = 'CJBSLWEVYK2LMFLPGRN1ROIXWKOQO1DLXWXCWQJN11NU3OA2' 
CLIENT_SECRET = 'OYZPNXPQMBE2OVBZSQ2G2R1XQRWH4BK3BCI3JHFTSEAOURGF' 
VERSION = '20180604'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[8]:


address = 'West Lawrence Avenue, Chicago'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print(latitude, longitude)


# <h3>Looking for the Italian Restaurants near the specified Location</h3>

# In[9]:


search_query = 'Restaurants'
radius = 5000
print(search_query + ' .... OK!')


# In[10]:


url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
url


# <b>Collect Inspiration Data</b>

# In[11]:


results = requests.get(url).json()
results


# In[ ]:


venues = results['response']['venues']

# tranform venues into a dataframe
dataframe = json_normalize(venues)
dataframe.head()


# <b>Explore and Understand Data</b>

# In[13]:


filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]


# In[14]:


dataframe_filtered


# In[15]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

dataframe_filtered


# <h3>Visualization of the nearby restaurants</h3>

# In[16]:


dataframe_filtered.name


# In[17]:


dataframe_filtered=dataframe_filtered.drop('formattedAddress', axis=1)
dataframe_filtered.head()


# In[18]:


venues_map = folium.Map(location=[latitude, longitude], zoom_start=13) # generate map centred around the West Lawrence Avenue

# add a red circle marker to represent the West Lawrence Avenue
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='West Lawrence Avenue',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# add the Italian restaurants as blue circle markers
for lat, lng, label in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)


# display map
venues_map


# <b>Data Modeling</b>

# In[19]:


df=dataframe_filtered[['distance','lat','lng']]
df.head()


# In[20]:


from sklearn.preprocessing import StandardScaler

X = df.values[:,1:]
X = np.nan_to_num(X)
cluster_dataset = StandardScaler().fit_transform(X)
cluster_dataset


# In[21]:


from sklearn.cluster import KMeans 
num_clusters = 3

k_means = KMeans(init="k-means++", n_clusters=num_clusters, n_init=12)
k_means.fit(cluster_dataset)
labels=k_means.labels_
labels


# In[22]:


dataframe_filtered["Labels"] = labels
dataframe_filtered.head(5)


# In[23]:


grouped_dataframe=dataframe_filtered.groupby('Labels').mean()
grouped_dataframe


# In[25]:


import matplotlib.cm as cm
import matplotlib.colors as colors
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(num_clusters)
ys = [i + x + (i*x)**2 for i in range(num_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lng, label,cluster in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories,dataframe_filtered.Labels):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# <h4>Cluster 1</h4>

# In[26]:


dataframe_filtered.loc[dataframe_filtered['Labels'] == 0, dataframe_filtered.columns[[1] + list(range(5, dataframe_filtered.shape[1]))]]


# <h3>Cluster 2</h3>

# In[27]:


dataframe_filtered.loc[dataframe_filtered['Labels'] == 1, dataframe_filtered.columns[[1] + list(range(5, dataframe_filtered.shape[1]))]]


# <h3>Cluster 3</h3>

# In[28]:


dataframe_filtered.loc[dataframe_filtered['Labels'] == 2, dataframe_filtered.columns[[1] + list(range(5, dataframe_filtered.shape[1]))]]


# In[ ]:




