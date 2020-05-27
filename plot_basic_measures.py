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

    ''' Plot basic measures in 5 subplots ordering cities for number of nodes '''
    measures_dict = nf.load_json(measures_file)
    df_measures = pd.DataFrame.from_dict(measures_dict).sort_values(by=['#nodes'], axis=1)
    fig = pf.plot_measures(df_measures)
    fig_name = prefix_png+'basic_measures/measures_plot.png'
    fig.savefig(fig_name)
    plt.close()

    ''' Plot measures against areas '''

    df_measures = pd.DataFrame.from_dict(measures_dict)
    df_area_population = pd.DataFrame.from_dict(area_population_dict)
    df = pd.concat([df_measures, df_area_population]).T
    x_values = [df['#nodes'], df['#edges'], df['density'], df['diameter'], df['avg_cc']]
    x_labels = list(df.columns.values)[:5]
    areas = df['area']
    markers = ['.', 'x', '+', 'o', '^']
    for x_val, x_label, i in zip(x_values, x_labels, range(len(x_values))):
        fig = pf.create_scatter(x_values=x_val, y_values=areas, x_label=x_label, y_label='area',
                                labels=df.index.values, colors=colors_cities)
        fig_name = prefix_png+'basic_measures/continents_division/'+x_label+'_continents.png'
        fig.savefig(fig_name)
    plt.close('all')

    ''' Plot x: #nodes, y: #edges and size of scatter points proportional to area of the city'''
    fig = pf.create_scatter(x_values=df['#nodes'], y_values=df['#edges'], x_label='number_of_nodes',
                            y_label='number_of_edges', labels=cities, colors=colors_cities, areas=df['area'])
    fig_name = prefix_png+'basic_measures/continents_division/nodes_vs_edges_continents_alpha_legends.png'
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close('all')

    ''' Plot #nodes/area on y axis and cities on x axis '''
    df = df.sort_values(by=['#nodes'])
    x_values = df.index.values
    y_values = [n / a for n, a in zip(df['#nodes'], df['area'])]

    plt.plot(x_values, y_values, marker='o')
    plt.xticks(rotation=70)
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel('#nodes/area')
    fig_name = prefix_png+'basic_measures/density_of_nodes.png'
    plt.savefig(fig_name, bbox_inches='tight')

    '''Plot CDF for areas and number of nodes and edges'''

    datavecs = [df['area'], df['#nodes'], df['#edges']]
    labels = ['area', '#nodes', '#edges']
    xlabel = 'measure'
    ylabel = 'P(measure)'
    fig = nf.plot_ccdf(datavecs, labels, xlabel, ylabel)
    # fig.show()
    fig_name = prefix_png+'basic_measures/ccdf_measures.png'
    fig.savefig(fig_name)

    ''' Plot degree distributions fo all cities '''
    degrees = nf.load_json(degrees_file)
    values = degrees.values()

    d = {}
    for city, list_k in degrees.items():
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
    df_degrees = pd.DataFrame.from_dict(d)

    ''' Plot all the degree distributions in one plot'''

    fig = nf.plot_multiple_distributions_with_colorbar_log_log_and_fitted_line(datavecs=degrees.values(),
                                                                               labels=cities, xlabel='k',
                                                                               ylabel='P(k)', c=areas, max_x=28)
    fig_name = 'results/all/plots/basic_measures/degree_distribution.png'
    fig.savefig(fig_name, bbox_inches='tight')

    clusters = nf.get_cluster_dict_for_area()
    for cluster, list_cities in clusters.items():
        datavecs, areas, max_k = [], [], 0
        for city in cities:
            if city in list_cities:
                max_city = int(df_degrees[city]['max(k)'])
                if max_city > max_k:
                    max_k = max_city
                datavecs.append(degrees[city])
                areas.append(area_population_dict[city]['area'])

        fig = nf.plot_multiple_distributions_with_colorbar_log_log_and_fitted_line(datavecs=datavecs,
                                                                                   labels=list_cities, xlabel='k',
                                                                                   ylabel='P(k)', c=areas, max_x=max_k)
        fig_name = prefix_png+'basic_measures/clusters/'+cluster+'_P(k).png'
        fig.savefig(fig_name, bbox_inches='tight')

    ''' Plot the degree distribution for each city and save the m of the fit line '''
    gammas = []
    for city, list_degrees in degrees.items():
        plt.close('all')
        fig, m = nf.plot_distribution_log_log(datavec=list_degrees, city=city, xlabel='k', ylabel='P(k)',
                                              max_x=int(df_degrees[city][1]))
        fig_name = 'results/'+city+'/degree_distribution.png'
        fig.savefig(fig_name, bbox_inches='tight')
        gammas.append(m)

    df_degrees = df_degrees.T
    df_degrees['\u03B3'] = gammas
    nf.dump_json('./results/all/json/gammas.txt', gammas)
    # print(df_degrees)
    # print(df_degrees.describe())
