import network_functions as nf
import generic_plot_functions as pf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    # ''' Load info about areas for each city '''
    # area_population_dict = nf.load_json(area_population_file)
    # areas = nf.get_list_sorted_values('area', area_population_dict)
    #
    # ''' Load info (previously computed) for certain measures for each network '''
    # measures_dict = nf.load_json(measures_file)
    # fig2 = pf.plot_measures(measures_dict)
    # fig_name = prefix_png+'basic_measures/measure_plot.png'
    # plt.savefig(fig_name)
    # plt.close()
    #
    # ''' Plot measures against areas '''
    # y_values = np.array(areas)
    # df = pd.DataFrame.from_dict(measures_dict)
    # x_values = list(df.to_numpy())
    # x_labels = list(df.index.values)
    # y_label = 'area'
    # labels = cities
    # markers = ['.', 'x', '+', 'o', '^']
    # for x_val, x_label, i in zip(x_values, x_labels, range(len(x_values))):
    #     fig = pf.create_scatter(x_val, y_values, x_label, y_label, labels, colors_cities)
    #     fig_name = prefix_png+'basic_measures/continents_division/'+x_label+'_continents.png'
    #     fig.savefig(fig_name)
    # plt.close('all')
    #
    # ''' Plot x: #nodes, y: #edges and size of scatter points proportional to area of the city'''
    # n_nodes = nf.get_list_sorted_values('#nodes', measures_dict)
    # x_values = np.array(n_nodes)
    # y_values = np.array(nf.get_list_sorted_values('#edges', measures_dict))
    # x_label = 'number_of_nodes'
    # y_label = 'number_of_edges'
    # labels = cities
    #
    # fig = pf.create_scatter(x_values, y_values, x_label, y_label, labels, colors_cities, areas)
    # fig_name = prefix_png+'basic_measures/continents_division/nodes_vs_edges_continents_alpha_legends.png'
    # fig.savefig(fig_name)
    # plt.close('all')
    #
    # '''Plot #nodes/area on y axis and cities on x axis'''
    # y_values = [n / a for n, a in zip(n_nodes, areas)]
    # x_values = cities
    #
    # plt.plot(x_values, y_values, marker='o')
    # plt.xticks(rotation=70)
    # plt.subplots_adjust(bottom=0.25)
    # plt.ylabel('#nodes/area')
    # fig_name = prefix_png+'density_of_nodes.png'
    # plt.savefig(fig_name)
    #
    # '''Plot CDF for areas and number of nodes and edges'''
    # n_edges = nf.get_list_sorted_values('#edges', measures_dict)
    # datavecs = [areas, n_nodes, n_edges]
    # labels = ['area', '#nodes', '#edges']
    # xlabel = 'measure'
    # ylabel = 'P(measure)'
    # fig = nf.plot_ccdf(datavecs, labels, xlabel, ylabel)
    # # fig.show()
    # fig_name = prefix_png+'ccdf_measures.png'
    # fig.savefig(fig_name)

    ''' Plot degree distributions fo all cities '''
    degrees = nf.load_json(degrees_file)
    values = degrees.values()
    # labels = cities
    # xlabel = 'Degree'
    # ylabel = 'P( Degree )'
    # fig = nf.plot_multiple_ccdf_with_colorbar(values, labels, xlabel, ylabel, areas)
    # fig_name = prefix_png+'ccdf_degree_distr.png'
    # fig.savefig(fig_name, bbox_inches='tight')

    # cols = ['City', 'min(k)', 'max(k)', '25%', '50%', '75%', '<k>']
    # df = pd.DataFrame(columns=cols)
    d = {}
    for city, list_k in degrees.items():

        row = {
            # 'City': city,
            'min(k)': np.min(list_k),
            'max(k)': np.max(list_k),
            '25%': np.percentile(list_k, 25),
            '50%': np.percentile(list_k, 50),
            '75%': np.percentile(list_k, 75),
            '<k>': float(f'{np.mean(list_k):.2f}')
        }
        d[city] = row
        # df.append(row, ignore_index=True)
    df = pd.DataFrame.from_dict(d)
    print(df.T)
