import network_functions as nf
import generic_plot_functions as pf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    cities = nf.get_list_cities_names()

    colors_cities = ['b', 'r', 'g', 'g', 'g', 'g', 'b', 'b', 'r', 'g', 'g', 'g', 'g', 'g', 'g', 'b', 'g', 'g',
                     'g', 'g', 'g', 'g', 'b', 'g', 'g', 'g', 'r']
    continent_colors = {'Oceania': 'b', 'America': 'r', 'Asia': 'y', 'Europe': 'g'}

    prefix_json = './results/all/json/'
    prefix_png = './results/all/plots/'

    measures_file = prefix_json+'measures.json'
    area_population_file = prefix_json+'area_population.json'
    degrees_file = prefix_json+'degrees.json'

    ''' Load info about areas for each city '''
    area_population_dict = nf.load_json(area_population_file)
    areas = nf.get_list_sorted_values('area', area_population_dict)

    ''' Load info (previously computed) for certain measures for each network '''
    measures_dict = nf.load_json(measures_file)
    fig2 = pf.plot_measures(measures_dict)
    fig_name = prefix_png+'basic_measures/measure_plot.png'
    plt.savefig(fig_name)
    plt.close()

    ''' Plot measures against areas '''
    y_values = np.array(areas)
    df = pd.DataFrame.from_dict(measures_dict)
    x_values = list(df.to_numpy())
    x_labels = list(df.index.values)
    y_label = 'area'
    labels = cities
    markers = ['.', 'x', '+', 'o', '^']
    for x_val, x_label, i in zip(x_values, x_labels, range(len(x_values))):
        fig = pf.create_scatter(x_val, y_values, x_label, y_label, labels, colors_cities)
        fig_name = prefix_png+'basic_measures/continents_division/'+x_label+'_continents.png'
        fig.savefig(fig_name)
    plt.close('all')

    ''' Plot x: #nodes, y: #edges and size of scatter points proportional to area of the city'''
    n_nodes = nf.get_list_sorted_values('#nodes', measures_dict)
    n_edges = nf.get_list_sorted_values('#edges', measures_dict)
    x_values = np.array(n_nodes)
    y_values = np.array(nf.get_list_sorted_values('#edges', measures_dict))
    x_label = 'number_of_nodes'
    y_label = 'number_of_edges'
    labels = cities

    fig = pf.create_scatter(x_values, y_values, x_label, y_label, labels, colors_cities, areas)
    fig_name = prefix_png+'basic_measures/continents_division/nodes_vs_edges_continents_alpha_legends.png'
    fig.savefig(fig_name)
    plt.close('all')

    '''Plot #nodes/area on y axis and cities on x axis'''
    y_values = [n / a for n, a in zip(n_nodes, areas)]
    x_values = cities

    plt.plot(x_values, y_values, marker='o')
    plt.xticks(rotation=70)
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel('#nodes/area')
    fig_name = prefix_png+'density_of_nodes.png'
    plt.savefig(fig_name)

    '''Plot CDF for areas and number of nodes and edges'''
    n_edges = nf.get_list_sorted_values('#edges', measures_dict)
    datavecs = [areas, n_nodes, n_edges]
    labels = ['area', '#nodes', '#edges']
    xlabel = 'measure'
    ylabel = 'P(measure)'
    fig = nf.plot_ccdf(datavecs, labels, xlabel, ylabel)
    # fig.show()
    fig_name = prefix_png+'ccdf_measures.png'
    fig.savefig(fig_name)

    ''' Plot degree distributions fo all cities '''
    degrees = nf.load_json(degrees_file)
    values = degrees.values()

    d = {}
    for city, E, N in zip(degrees.keys(), n_edges, n_nodes):
        list_k = degrees[city]
        row = {
            'min(k)': np.min(list_k),
            'max(k)': np.max(list_k),
            '25%': np.percentile(list_k, 25),
            '50%': np.percentile(list_k, 50),
            '75%': np.percentile(list_k, 75),
            '<k>': float(f'{np.mean(list_k):.2f}')
        }
        d[city] = row
    pd.set_option('precision', 2)
    df = pd.DataFrame.from_dict(d).T
    # print(df)
    # print(df[df['max(k)'] == df['max(k)'].max()])

    ''' Plot all the degree distributions in one plot'''
    labels = cities
    xlabel = 'k'
    ylabel = 'P(k)'
    datavecs = degrees.values()
    areas = nf.get_list_sorted_values('area', area_population_dict)
    fig = nf.plot_multiple_distributions_with_colorbar(datavecs, labels, xlabel, ylabel, areas)
    # fig.show()
    fig_name = 'results/all/plots/basic_measures/degree_distribution.png'
    fig.savefig(fig_name, bbox_inches='tight')

    # clusters = nf.get_cluster_dict_for_area()
    # for cluster, list_cities in clusters.items():
    #     datavecs = [list_k for city, list_k in degrees.items() if city in list_cities]
    #     area_dict = {k: v for k, v in area_population_dict.items() if k in list_cities}
    #     areas = nf.get_list_sorted_values('area', area_dict)
    #     fig = nf.plot_multiple_distributions_with_colorbar(datavecs, list_cities, xlabel, ylabel, areas)
    #     fig.show()

    ''' Plot the degree distribution for each city and save the m of the fit line '''
    gammas = []
    for city, list_degrees in degrees.items():
        # print(df)
        max_k = int(df.T[city][1])
        xlabel = 'k'
        ylabel = 'P(k)'
        plt.close('all')
        fig, m = nf.plot_degree_distribution(list_degrees, city, xlabel, ylabel, max_k)
        # fig.show()
        fig_name = 'results/'+city+'/degree_distribution.png'
        fig.savefig(fig_name, bbox_inches='tight')
        gammas.append(m)

    df['\u03B3'] = gammas
    print(df)
