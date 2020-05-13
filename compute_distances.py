import network_functions as nf
import networkx as nx
import random as rd


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types = nf.get_types_of_transport_and_colors()

    distances_info = {'bfs': [], 'eu': []}

    n_core_nodes = 20
    dump = False  # put True if you want to save again values into json file

    # all_bfs, all_ed = [], []  # list of lists of values for each city

    ''' Computations over each city '''
    for city in cities:

        print('Processing ' + city + ' ...')

        ''' CREATE NETWORK - undirected and unweighted and add it to the proper dictionary '''
        net = nf.create_network(city, types=types)
        nodes = net.nodes()

        city_bfs, city_ed = [], []

        max_component = max(nx.connected_component_subgraphs(net), key=len)

        for i in range(n_core_nodes):

            core_node = rd.choice(list(max_component.nodes()))
            # print("Core node = %d" % core_node)
            coords = nx.get_node_attributes(net, 'coords')

            # for each core node visit the network and record 3 types of distances for each reached node:
            #   1. bfs
            #   2. euclidean distance
            # all in km
            level, parent, distances_bfs, distances_eu = nf.bfs_with_distance(net, core_node, coords)

            # bfs, ed = nf.get_all_distances(net, core_node)
            # concatenate those list to city ones
            city_bfs.extend(distances_bfs.values())
            city_ed.extend(distances_eu.values())

        distances_info['bfs'].append(city_bfs)
        distances_info['eu'].append(city_ed)

    if dump:
        nf.dump_json('results/all/json/all_distances.json', distances_info)

