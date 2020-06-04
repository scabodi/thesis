import network_functions as nf
import matplotlib.pyplot as plt


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    clusters = nf.get_cluster_dict_for_area()

    distances_file = 'results/all/json/all_distances.json'
    dist_dict = nf.load_json(distances_file)

    bfs_random = nf.load_json('results/all/json/bfs_all.json')
    eu_random = nf.load_json('results/all/json/eu_all.json')

    area_population_file = 'results/all/json/area_population.json'
    area_population_dict = nf.load_json(area_population_file)
    areas = nf.get_list_sorted_values('area', area_population_dict)

    x_values = cities
    y_values = []

    fractions = {}
    mu_st = {}

    for i, city in enumerate(cities, 0):

        ''' Distances distributions: two histograms with info about mean and standard deviation '''
        bfs_list = [value for value in bfs_random[i] if value != 0]
        eu_list = [value for value in eu_random[i] if value != 0]

        # create a third array with the fractions
        f = [x/y for (x, y) in zip(eu_list, bfs_list)]
        fractions[city] = f

        # TODO collect info about mean and st dev
        # plot both distribution for the single city in one p the lot
        fig, params = nf.plot_distances_for_single_city(bfs_list, eu_list)
        fig_name = './results/'+city+'/distances_histo.png'
        plt.savefig(fig_name)

        mu_st[city] = params

        ''' Close and far nodes distribution from central node '''

        ''' Cumulative distribution of distances from central node '''

        ''' Network plot of periferal nodes '''

    for cluster, values in clusters.items():

        f = {k: v for k, v in fractions.items() if k in values}
        fig = nf.plot_distances_fractions(f)
        fig_name = './results/all/plots/fractions_'+cluster+'_.png'
        fig.savefig(fig_name, bbox_inches='tight')
