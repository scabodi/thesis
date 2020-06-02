# new version of compute_distances.py

import network_functions as nf
import networkx as nx
import random as rd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types = nf.get_types_of_transport_and_colors()

    distances_info = {'bfs': [], 'eu': []}

    n_core_nodes = 20
    dump = False  # put True if you want to save again values into json file

    # all_bfs, all_ed = [], []  # list of lists of values for each city

    ''' Computations over each city '''
    for city in cities[4:]:

        print('Processing ' + city + ' ...')

        ''' CREATE NETWORK - undirected and unweighted '''
        net = nf.create_network(city, types=types)
        nodes = net.nodes()

        city_bfs, city_ed = [], []

        max_component = max(nx.connected_component_subgraphs(net), key=len)
        coords = nx.get_node_attributes(max_component, 'pos')
        a = np.array(list(coords.values()))
        centre = centeroidnp(a)

        nearest = min(a, key=lambda x: distance.euclidean(x, centre))
        central_node = None
        for n, coord in coords.items():
            if coord == tuple(nearest):
                central_node = n
                break
        print(central_node)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        nx.draw_networkx(max_component, ax=ax, pos=coords, with_labels=False,
                         node_size=50, alpha=0.5, nodelist=[central_node, 146])
        fig.show()

        for i in range(n_core_nodes):

            core_node = rd.choice(list(max_component.nodes()))
            # print("Core node = %d" % core_node)
            coords = nx.get_node_attributes(net, 'coords')

            ''' 
            for each core node visit the network and record 2 types of distances for each reached node:
                1. bfs - minimum hop not minimum distance 
                2. euclidean distance
            all in km 
            maybe visit a weighted graph with dijskstra
            '''
            level, parent, distances_bfs, distances_eu = nf.bfs_with_distance(net, core_node, coords)

            nodes = distances_bfs.keys()
            bfs_list = distances_bfs.values()
            eu_list = distances_eu.values()

            max_eu = max(eu_list)
            threshold_near = 1/4*max_eu
            threshold_far = 1/2*max_eu  # vedere 3/4

            close_distances, far_distances = {}, {}

            for node, bfs, eu in zip(nodes, bfs_list, eu_list):
                if bfs < threshold_near:
                    close_distances[node] = bfs
                elif bfs > threshold_far:
                    far_distances[node] = bfs

            print(close_distances)
            print(far_distances)

            # bfs, ed = nf.get_all_distances(net, core_node)
            # concatenate those list to city ones
            city_bfs.extend(distances_bfs.values())
            city_ed.extend(distances_eu.values())

        distances_info['bfs'].append(city_bfs)
        distances_info['eu'].append(city_ed)

    if dump:
        nf.dump_json('results/all/json/all_distances.json', distances_info)

