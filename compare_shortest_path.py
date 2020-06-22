import network_functions as nf
import networkx as nx
import random as rd
import os


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types = nf.get_types_of_transport_and_colors()
    clusters = nf.get_cluster_dict_for_area()

    n_core_nodes = 20
    dump = False

    fractions = {} if dump is True else nf.load_json('./results/all/json/fractions_bfs_vs_sp.json')

    if dump:
        ''' Computations over each city '''
        for city in cities:

            print('Processing ' + city + ' ...')

            ''' CREATE NETWORK - undirected and unweighted '''
            net = nf.create_network(city, types=types)
            net_w = net.copy()
            nf.add_weights_to_network(net_w)
            nodes = net.nodes()

            city_bfs, city_ed = [], []

            f = []

            max_component = max(nx.connected_component_subgraphs(net), key=len)

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
                r = distances_bfs.pop(core_node, None)

                sp = nx.shortest_path_length(net_w, source=core_node, weight='weight')
                r1 = sp.pop(core_node, None)

                bfs_nodes = set(distances_bfs.keys())
                sp_nodes = set(sp.keys())
                equal = True if bfs_nodes == sp_nodes else False

                for node in bfs_nodes:
                    f.append(sp[node]/distances_bfs[node])

            fractions[city] = f

        nf.dump_json('./results/all/json/fractions_bfs_vs_sp.json', fractions)

    for cluster, values in clusters.items():

        f = {k: v for k, v in fractions.items() if k in values}
        fig = nf.plot_distances_fractions(f)
        dir_name = './results/all/plots/distances/fractions_sp/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig_name = dir_name+'fractions_'+cluster+'_.png'
        fig.savefig(fig_name, bbox_inches='tight')

    fractions = nf.load_json('./results/all/json/fractions_bfs_vs_sp.json')

    fig = nf.plot_distances_fractions(fractions, False)
    fig_name = './results/all/plots/distances/fractions_sp/all_no_legend.png'
    fig.savefig(fig_name, bbox_inches='tight')
