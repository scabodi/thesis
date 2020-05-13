import network_functions as nf
import matplotlib.pyplot as plt


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    clusters = nf.get_cluster_dict_for_area()

    distances_file = 'results/all/json/all_distances.json'
    dist_dict = nf.load_json(distances_file)

    area_population_file = 'results/all/json/area_population.json'
    area_population_dict = nf.load_json(area_population_file)
    areas = nf.get_list_sorted_values('area', area_population_dict)

    '''Plot all distributions in 2 different plots, one for BFS and the other one for euclidean'''
    datavecs = dist_dict['bfs']
    labels = cities
    xlabel = 'Distance'
    ylabel = 'P(distance)'
    c = areas

    fig1 = nf.plot_multiple_ccdf_with_colorbar(datavecs, labels, xlabel, ylabel, c)
    fig_name = './results/all/plots/distances/bfs.png'
    fig1.savefig(fig_name)

    datavecs = dist_dict['eu']
    fig2 = nf.plot_multiple_ccdf_with_colorbar(datavecs, labels, xlabel, ylabel, c)
    fig_name = './results/all/plots/distances/eu.png'
    fig2.savefig(fig_name)

    x_values = cities
    y_values = []

    fractions = {}

    for i, city in enumerate(cities, 0):

        bfs_list = [value for value in dist_dict['bfs'][i] if value != 0]
        eu_list = [value for value in dist_dict['eu'][i] if value != 0]

        # create a third array with the fractions
        f = [x/y for (x, y) in zip(eu_list, bfs_list)]
        fractions[city] = f

        # plot both distribution for the single city in one p the lot
        fig = nf.plot_distances_for_single_city(bfs_list, eu_list)
        fig_name = './results/'+city+'/distances_histo.png'
        plt.savefig(fig_name)

    for cluster, values in clusters.items():

        f = {k: v for k, v in fractions.items() if k in values}
        fig = nf.plot_distances_fractions(f)
        fig_name = './results/all/plots/fractions_'+cluster+'_.png'
        fig.savefig(fig_name, bbox_inches='tight')
