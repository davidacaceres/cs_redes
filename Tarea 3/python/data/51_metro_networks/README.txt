"A curated data set of L-space representations in .json format for 51 metro networks worldwide."

The goal of this data set was to create a set of topological representations based on GTFS data that is both curated and verified. 
Each file is a graph, corresponding to a network, containing a set of nodes and edges for each network along with some additional information. 
The information on the number of vehicles is based on a representative day for the period of 05:00 until 23:59. 
The representative day chosen for each network along with information about the processing can be found in metadata.xlsx 

These representations were created using a Python pipeline that processes the GTFS files into a .json format that can be read by the NetworkX library.
The primary library used for processing is gtfspy[1] which turns the GTFS files into the .sqlite format and an initial network representation.
The pipeline was then used to turn this initial representation into proper L- and P-space representations. 
The exact decisions in the implementation can be found in the metadata.xlsx file. 
In order to use this data, a Python script should be created that uses NetworkX to read the json data into the NetworkX format[2].
This NetworkX representation can then be visualized using for example the Bokeh library.
For further details on the creation process of these representations, please refer to the report referenced in the description.

Data format:

"directed": 0 if undirected, 1 if directed. For this dataset, all edges are directed.

"multigraph": False. The graph is not a multigraph, each edge exists only once.

"graph": Left intentionally empty

"nodes":
	"name": The station name
	"lat": The latitude of the station
	"lon": The longitude of the station
	"id": A unique id to identify each station
	"original_ids": Some stations have been merged in order to properly represent transfer possibilities. The information about the original nodes is retained via original_ids.

"links":
	"duration_avg": The average in-vehicle travel time for this edge (seconds)
	"n_of_vehicles": The total number of vehicles crossing this edge in the given time period (05:00-23:59)
	"d": The direct distance between the two stations (meters)
	"route_I_counts": The number of vehicles split per route crossing this edge in the given time period (05:00-23:59)
	"shape_id*": The number of vehicles split per shape_id crossing this edge in the given time period (05:00-23:59)
	"direction_id"*: The number of vehicles split per direction crossing this edge in the given time period (05:00-23:59)
	"headsign"*: The number of vehicles split per headsign crossing this edge in the given time period (05:00-23:59)
	"source": The source node
	"target": The target node


* optional


[1] Rainer Kujala, Christoffer Weckström, Miloš N. Mladenović, Jari Saramäki, 
Travel times and transfers in public transport: Comprehensive accessibility analysis based on Pareto-optimal journeys, 
In Computers, Environment and Urban Systems, Volume 67, 2018, Pages 41-54, ISSN 0198-9715, 
https://doi.org/10.1016/j.compenvurbsys.2017.08.012.

[2] https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_graph.html#networkx.readwrite.json_graph.node_link_graph