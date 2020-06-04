# new version of compute_distances.py
import network_functions as nf
import networkx as nx
import random as rd
import os


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    capitals = nf.get_capitals_with_central_station_node()
    types = nf.get_types_of_transport_and_colors()

    prefix_json = 'results/all/json/'
    bfs_json = prefix_json+'bfs_all.json'
    eu_json = prefix_json+'eu_all.json'
    central_nodes_json = prefix_json+'central_nodes.json'

    distances_info = {'bfs': [], 'eu': []}
    central_nodes = {}

    n_core_nodes = 20
    dump = False  # put True if you want to save again values into json file

    # all_bfs, all_ed = [], []  # list of lists of values for each city

    ''' Computations over each city '''
    for city in cities:

        print('Processing ' + city + ' ...')

        ''' Create directory for distance analysis in the city result folder '''
        dir_distance = './results/' + city + '/distance_analysis/'
        dir_json = './results/' + city + '/distance_analysis/json/'
        if not os.path.exists(dir_distance):
            os.makedirs(dir_distance)
        if not os.path.exists(dir_json):
            os.makedirs(dir_json)

        bfs_close_json = dir_json + 'bfs_close.json'
        bfs_far_json = dir_json + 'bfs_far.json'
        bfs_central = dir_json + 'bfs_central.json'
        eu_central = dir_json + 'eu_central.json'

        ''' CREATE NETWORK - undirected and unweighted '''
        net = nf.create_network(city, types=types)
        nodes = net.nodes()

        city_bfs, city_ed = [], []

        max_component = max(nx.connected_component_subgraphs(net), key=len)
        coords = nx.get_node_attributes(max_component, 'pos')
        coords_for_geopy = nx.get_node_attributes(net, 'coords')

        ''' Breadth first visit of the graph starting from a random node - repeated 20 times '''
        for i in range(n_core_nodes):

            core_node = rd.choice(list(max_component.nodes()))
            # print("Core node = %d" % core_node)
            '''
            for each core node visit the network and record 2 types of distances for each reached node:
                1. bfs - minimum hop not minimum distance
                2. euclidean distance
            all in km
            maybe visit a weighted graph with dijskstra
            '''
            level, parent, distances_bfs, distances_eu = nf.bfs_with_distance(net, core_node, coords_for_geopy)

            # bfs, ed = nf.get_all_distances(net, core_node)
            # concatenate those list to city ones
            city_bfs.extend(distances_bfs.values())
            city_ed.extend(distances_eu.values())

        distances_info['bfs'].append(city_bfs)
        distances_info['eu'].append(city_ed)

        central_node = nf.get_central_node(coords)[city]
        central_nodes[city] = central_node

        ''' Breadth first visit of the graph starting from the central node  '''
        level, parent, distances_bfs, distances_eu = nf.bfs_with_distance(net, central_node, coords_for_geopy)
        close_distances, far_distances = nf.compute_near_and_far_distances_dictionaries(nodes=distances_bfs.keys(),
                                                                                        bfs_list=distances_bfs.values(),
                                                                                        eu_list=distances_eu.values())
        '''find periferal nodes '''
        periferal_dict = {node: True for node in distances_eu.keys()}

        for edge in net.edges():
            if edge[0] in periferal_dict and edge[1] in periferal_dict:
                # a, b = int(edge[0]), int(edge[1])
                if distances_eu[edge[0]] > distances_eu[edge[1]]:
                    periferal_dict[edge[1]] = False
                if distances_eu[edge[0]] < distances_eu[edge[1]]:
                    periferal_dict[edge[0]] = False

        if dump:
            nf.dump_json(bfs_central, distances_bfs)
            nf.dump_json(eu_central, distances_eu)
            nf.dump_json(bfs_close_json, close_distances)
            nf.dump_json(bfs_far_json, far_distances)
            nf.dump_json(dir_json + 'periferal.json', periferal_dict)

        if city in capitals:
            level, parent, distances_bfs, distances_eu = nf.bfs_with_distance(net, capitals[city], coords_for_geopy)
            close_distances, far_distances = nf.compute_near_and_far_distances_dictionaries(nodes=distances_bfs.keys(),
                                                                                            bfs_list=
                                                                                            distances_bfs.values(),
                                                                                            eu_list=
                                                                                            distances_eu.values())
            if dump:
                nf.dump_json(dir_json + 'capital_bfs_central.json', distances_bfs)
                nf.dump_json(dir_json + 'capital_eu_central.json', distances_eu)
                nf.dump_json(dir_json + 'capital_bfs_close.json', close_distances)
                nf.dump_json(dir_json + 'capital_bfs_far.json', far_distances)

    if dump:
        nf.dump_json(bfs_json, distances_info['bfs'])
        nf.dump_json(eu_json, distances_info['eu'])
        nf.dump_json(central_nodes_json, central_nodes)
