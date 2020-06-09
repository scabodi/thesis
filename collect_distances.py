# new version of compute_distances.py
import network_functions as nf
import networkx as nx
import random as rd
import os
from geopy.distance import geodesic
import pandas as pd
import numpy as np


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

    df = pd.DataFrame(columns=['City', 'N', 'comp 1', 'comp 2', 'comp 3', 'comp 4', 'n comp'])
    df['City'] = cities
    df.set_index("City", inplace=True)

    # all_bfs, all_ed = [], []  # list of lists of values for each city

    ''' Computations over each city '''
    for city in cities:

        print('Processing ' + city + ' ...')

        ''' Create directory for distance analysis in the city result folder '''
        dir_json = './results/' + city + '/distance_analysis/json/'
        dir_components = './results/' + city + '/connected_components/plots/'
        if not os.path.exists(dir_components):
            os.makedirs(dir_components)
        if not os.path.exists(dir_json):
            os.makedirs(dir_json)

        bfs_close_json = dir_json + 'bfs_close.json'
        bfs_far_json = dir_json + 'bfs_far.json'
        bfs_central = dir_json + 'bfs_central.json'
        eu_central = dir_json + 'eu_central.json'

        ''' CREATE NETWORK - undirected and unweighted '''
        net = nf.create_network(city, types=types)
        nodes = net.nodes()
        n_nodes = len(nodes)

        city_bfs, city_ed = [], []

        ''' Analysis on connected components '''
        df.loc[[city], ['N']] = n_nodes
        connected_components = sorted(nx.connected_components(net), key=len, reverse=True)
        df.loc[[city], ['n comp']] = len(connected_components)
        for i, cc in enumerate(connected_components, 1):
            sub_net = net.subgraph(cc)
            fig, ax = nf.plot_network(sub_net)
            n_nodes_cc = len(cc)
            if 'comp '+str(i) in df.columns:
                df.loc[[city], ['comp '+str(i)]] = n_nodes_cc
            ax.set_title(city+' component with %d nodes out of %d' % (n_nodes_cc, n_nodes))
            fig.savefig(dir_components+'component_%d' % i, bbox_inches='tight')

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

        '''find peripheral nodes '''
        peripheral_dict = nf.get_peripheral_nodes(net=net, coords=coords_for_geopy, distances_eu=distances_eu,
                                                  json_file=dir_json + 'peripheral_nodes.json')
        if dump:
            nf.dump_json(bfs_central, distances_bfs)
            nf.dump_json(eu_central, distances_eu)
            nf.dump_json(bfs_close_json, close_distances)
            nf.dump_json(bfs_far_json, far_distances)
            nf.dump_json(dir_json + 'peripheral.json', peripheral_dict)

        if city in capitals:
            ''' Plot geo central node and POI central node '''
            # fig, _ = nf.plot_network(net, node_list=[central_node, capitals[city]])
            # fig.show()
            # bool = nx.has_path(net, central_node, capitals[city])
            #
            # print('Number of nodes reached by geo centre: %d' % (len(distances_bfs)))
            level, parent, distances_bfs, distances_eu = nf.bfs_with_distance(net, capitals[city], coords_for_geopy)
            close_distances, far_distances = nf.compute_near_and_far_distances_dictionaries(
                nodes=distances_bfs.keys(), bfs_list=distances_bfs.values(), eu_list=distances_eu.values())
            # print('Number of nodes reached by POI centre: %d' % (len(distances_bfs)))

            if dump:
                nf.dump_json(dir_json + 'capital_bfs_central.json', distances_bfs)
                nf.dump_json(dir_json + 'capital_eu_central.json', distances_eu)
                nf.dump_json(dir_json + 'capital_bfs_close.json', close_distances)
                nf.dump_json(dir_json + 'capital_bfs_far.json', far_distances)

    if dump:
        nf.dump_json(bfs_json, distances_info['bfs'])
        nf.dump_json(eu_json, distances_info['eu'])
        nf.dump_json(central_nodes_json, central_nodes)

    print(df)
    df1 = df.replace(np.nan, '', regex=True)
    df1.style.set_properties(**{'text-align': 'center'})
    html_string = nf.get_html_string()
    with open('./results/all/tables/components_html.html', 'w') as f:
        f.write(html_string.format(table=df1.to_html(classes='mystyle')))
