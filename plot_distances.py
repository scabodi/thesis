import network_functions as nf
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    capitals = nf.get_capital_cities()
    clusters = nf.get_cluster_dict_for_area()

    bfs_random = nf.load_json('results/all/json/bfs_all.json')
    eu_random = nf.load_json('results/all/json/eu_all.json')

    area_population_file = 'results/all/json/area_population.json'
    area_population_dict = nf.load_json(area_population_file)
    areas = nf.get_list_sorted_values('area', area_population_dict)

    fractions = {}
    mu_st = {}

    for i, city in enumerate(cities, 0):

        print('Processing ' + city + ' ...')

        dir_plots = './results/' + city + '/distance_analysis/plots/'
        if not os.path.exists(dir_plots):
            os.makedirs(dir_plots)

        ''' Distances distributions: two histograms with info about mean and standard deviation '''
        bfs_list = [value for value in bfs_random[i] if value != 0]
        eu_list = [value for value in eu_random[i] if value != 0]

        # create a third array with the fractions
        f = [x/y for (x, y) in zip(eu_list, bfs_list)]
        fractions[city] = f

        # plot both distribution for the single city in one p the lot
        colors = ['skyblue', 'red']
        bins = [int(np.sqrt(len(bfs_list))/10), int(np.sqrt(len(eu_list))/10)]
        fig, params = nf.plot_distances_for_single_city(first=bfs_list, second=eu_list, colors=colors,
                                                        labels=["bfs", "euclidean"], bins=bins,
                                                        xlabel='Distance in km from start node', ylabel='P(distance)')
        # fig.show()
        fig_name = dir_plots+'distances_histo.png'
        fig.savefig(fig_name)

        mu_st[city] = params

        ''' Close and far nodes distribution from central node '''
        close_distances = nf.load_json('results/'+city+'/distance_analysis/json/bfs_close.json')
        far_distances = nf.load_json('results/'+city+'/distance_analysis/json/bfs_far.json')

        first = list(close_distances.values())
        second = list(far_distances.values())
        bins = [int(np.sqrt(len(first))), int(np.sqrt(len(second)))]
        fig1, _ = nf.plot_distances_for_single_city(first=first, second=second, colors=colors,
                                                    labels=["close nodes", "far nodes"], bins=bins,
                                                    xlabel='Distance in km from central node', ylabel='P(distance)')
        fig_name = dir_plots+'close_far_distances.png'
        fig1.savefig(fig_name)

        ''' Plot 1-CDF curve of distances from central node '''
        central_bfs = nf.load_json('results/'+city+'/distance_analysis/json/bfs_central.json')
        central_eu = nf.load_json('results/'+city+'/distance_analysis/json/eu_central.json')
        datavecs = [list(central_bfs.values()), list(central_eu.values())]
        fig, _ = nf.plot_ccdf(datavecs=datavecs, labels=['bfs', 'euclidean'], xlabel='Distance in km', ylabel='1-CDF',
                              marker='^')
        fig_name = dir_plots+'ccdf_central.png'
        fig.savefig(fig_name)

        ''' Same plot but from POI node - just for capitals '''
        if city in capitals:
            close_distances = nf.load_json('results/' + city + '/distance_analysis/json/capital_bfs_close.json')
            far_distances = nf.load_json('results/' + city + '/distance_analysis/json/capital_bfs_far.json')

            first = list(close_distances.values())
            second = list(far_distances.values())
            bins = [int(np.sqrt(len(first))), int(np.sqrt(len(second)))]
            fig1, _ = nf.plot_distances_for_single_city(first=first, second=second, colors=colors,
                                                        labels=["close nodes", "far nodes"], bins=bins,
                                                        xlabel='Distance in km from POI central node',
                                                        ylabel='P(distance)')

            # fig1.show()
            fig_name = dir_plots + 'capital_close_far_distances.png'
            fig1.savefig(fig_name)

            central_bfs_capital = nf.load_json('results/' + city + '/distance_analysis/json/capital_bfs_central.json')
            central_eu_capital = nf.load_json('results/' + city + '/distance_analysis/json/capital_eu_central.json')
            datavecs = [list(central_bfs_capital.values()), list(central_eu_capital.values())]
            fig, _ = nf.plot_ccdf(datavecs=datavecs, labels=['bfs', 'euclidean'], xlabel='Distance in km',
                                  ylabel='1-CDF', marker='^')
            fig_name = dir_plots + 'ccdf_central_POI.png'
            fig.savefig(fig_name)

        plt.close('all')

        ''' Cumulative distribution of distances from central node '''

        # ''' Network plot of periferal nodes '''
        # peripheral_dict = nf.load_json('results/'+city+'/distance_analysis/json/peripheral.json')
        # peripheral_nodes = [int(node) for node, is_peripheral in peripheral_dict.items() if is_peripheral is True]
        #
        # net = nf.create_network(city, types=nf.get_types_of_transport_and_colors())
        # percentage = (len(peripheral_nodes) / len(net.nodes()))*100
        # print('There are %d peripheral nodes.\n They represent %d %% of all the nodes.'
        #       % (len(peripheral_nodes), percentage))
        # fig = nf.plot_network(net, node_list=peripheral_nodes)
        # fig.show()
        # fig_name = dir_plots+'network_peripheral.png'
        # fig.savefig(fig_name)

    # nf.dump_json('results/all/json/parameter_distances.json', mu_st)
    #
    # for cluster, values in clusters.items():
    #
    #     f = {k: v for k, v in fractions.items() if k in values}
    #     fig = nf.plot_distances_fractions(f)
    #     dir_name = './results/all/plots/distances/fractions/'
    #     if not os.path.exists(dir_name):
    #         os.makedirs(dir_name)
    #     fig_name = dir_name+'fractions_'+cluster+'_.png'
    #     fig.savefig(fig_name, bbox_inches='tight')
