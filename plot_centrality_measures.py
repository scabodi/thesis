import network_functions as nf
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types = nf.get_types_of_transport_and_colors()

    centrality_dict = {'degree': [], 'betweenness': [], 'closeness': [], 'eigenvector': []}

    area_population_file = 'results/all/json/area_population.json'
    area_population_dict = nf.load_json(area_population_file)
    areas = nf.get_list_sorted_values('area', area_population_dict)
    populations = nf.get_list_sorted_values('population', area_population_dict)

    ''' Computations over each city '''
    for city in cities:

        print('Processing ' + city + ' ...')

        net = nf.create_network(city, types=types)
        nf.plot_network(city, net)

        ''' Load centrality measures for specific city '''
        json_path = 'results/'+city+'/centrality_measures.json'
        centrality_measures = nf.load_json(json_path)
        for k, v in centrality_measures.items():
            centrality_dict[k].append(v)

        ''' Plot ccdf of all measures of centrality considered for current city '''
        fig = nf.plot_ccdf(datavecs=list(centrality_measures.values()), labels=list(centrality_measures.keys()),
                           xlabel='measure', ylabel='P(measure)')
        dir_name = './results/'+city+'/centrality_measures/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig_name = dir_name + 'ccdf_centrality_measures.png'
        fig.savefig(fig_name)

        ''' Plot all distributions together '''
        fig = nf.plot_multiple_lines(x_values=net.nodes(), y_values=list(centrality_measures.values()),
                                     labels=list(centrality_measures.keys()), xlabel='measure', ylabel='P(measure)')
        dir_name = './results/' + city + '/centrality_measures/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig_name = dir_name+'all_centrality_distributions.png'
        fig.savefig(fig_name)

        for k, v in centrality_measures.items():
            sorted_values = v.copy()
            sorted_values.sort()

            ''' Plot distribution of current centrality measure'''
            fig1 = nf.plot_distribution(list(net.nodes()), list(sorted_values), "nodes", k)
            dir_name = './results/'+city+'/centrality_measures/distributions/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig_name = dir_name+'distr_plot_'+k+'_centrality.png'
            fig1.savefig(fig_name)

            ''' Plot network with 20% of significant nodes with color depending on value of centrality '''
            fig2 = nf.plot_network_with_node_color_based_on_measure(net=net, measures=v, title=k+' centrality')
            dir_name = './results/'+city+'/centrality_measures/network/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig_name = dir_name+'network_'+k+'_centrality.png'
            fig2.savefig(fig_name)
            plt.close('all')

# print(centrality_measures)
# df = pd.DataFrame.from_dict(centrality_measures)
# print(df)
# values = [np.array(v) for k, v in centrality_measures.items()]
#
# # Save in the dict for each type of centrality a list of lists of values to plot later
# for k, v in centrality_measures.items():
#     if k not in centrality_dict:
#         centrality_dict[k] = []
#     centrality_dict[k].append(list(v))

''' Plot all distributions of the same centrality type for all the cities coloring with colormap the 
line depending on the area of the city and the population of it  '''
# for measure, values in centrality_dict.items():
#     xlabel = measure
#     ylabel = 'P('+measure+')'
#     labels = cities
#     fig = nf.plot_multiple_ccdf_with_colorbar(values, labels, xlabel, ylabel, c=areas)
#     fig_name = './results/all/plots/centrality_measures/ccdf_'+measure+'_centrality_area.png'
#     # fig.show()
#     fig.savefig(fig_name, bbox_inches='tight')
#
#     fig = nf.plot_multiple_ccdf_with_colorbar(values, labels, xlabel, ylabel, c=populations)
#     fig_name = './results/all/plots/centrality_measures/ccdf_' + measure + '_centrality_population.png'
#     # fig.show()
#     fig.savefig(fig_name, bbox_inches='tight')



