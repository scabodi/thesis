import network_functions as nf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types = nf.get_types_of_transport_and_colors()

    degree_betweenness = nf.load_json('results/all/json/degree_betweenness.json')

    centralities_path = './results/all/json/centrality_measures.json'

    centrality_dict = {'degree': [], 'betweenness': [], 'closeness': [], 'eigenvector': []}

    area_population_file = 'results/all/json/area_population.json'
    area_population_dict = nf.load_json(area_population_file)
    areas = nf.get_list_sorted_values('area', area_population_dict)
    populations = nf.get_list_sorted_values('population', area_population_dict)

    lambdas, etas = [], []
    ''' Computations over each city '''
    for city in cities:

        print('Processing ' + city + ' ...')

        net = nf.create_network(city, types=types)
        fig, _ = nf.plot_network(net)
        fig_name = './results/' + city + '/network.png'
        fig.savefig(fig_name)

        ''' Load centrality measures for specific city '''
        json_path = 'results/'+city+'/centrality_measures/json/centrality_measures.json'
        centrality_measures = nf.load_json(json_path)
        for k, v in centrality_measures.items():
            centrality_dict[k].append(v)

        ''' Scatter plots: betweenness vs closeness and betweenness vs eigenvector'''
        df_centralities = pd.DataFrame.from_dict(centrality_measures)
        fig, ax = nf.plot_two_columns_dataframe(df=df_centralities, col_x='betweenness', col_y1='closeness')
        ax.set_title(city)
        fig.show()
        dir_scatter = './results/' + city + '/centrality_measures/scatter/'
        if not os.path.exists(dir_scatter):
            os.makedirs(dir_scatter)
        fig_name = dir_scatter + 'betweenness_vs_closeness.png'
        fig.savefig(fig_name)

        fig, ax = nf.plot_two_columns_dataframe(df=df_centralities, col_x='betweenness', col_y1='eigenvector')
        ax.set_title(city)
        fig.show()
        fig_name = dir_scatter + 'betweenness_vs_eigenvector.png'
        fig.savefig(fig_name)

        ''' Plot ccdf of all measures of centrality considered for current city '''
        fig, _ = nf.plot_ccdf(datavecs=list(centrality_measures.values()), labels=list(centrality_measures.keys()),
                              xlabel='measure', ylabel='P(measure)')
        # fig.show()
        dir_name = './results/'+city+'/centrality_measures/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig_name = dir_name + 'ccdf_centrality_measures.png'
        fig.savefig(fig_name)

        plt.close('all')
        ''' Plot correlation between degree and average betweenness centrality '''
        fig, m = nf.plot_correlation_measures_log_log(x_values=[int(x) for x in degree_betweenness[city].keys()],
                                                      y_values=list(degree_betweenness[city].values()), xlabel='k',
                                                      ylabel='<b>', title=city)
        etas.append(m)
        dir_name = './results/' + city + '/centrality_measures/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig_name = dir_name + 'degree_betweenness_correlation.png'
        fig.savefig(fig_name, bbox_inches='tight')

        for measure, values in centrality_measures.items():
            plt.close('all')

            ''' Plot distribution of current centrality measure'''
            fig1, m = nf.plot_ccdf(datavecs=[values], labels=[measure], xlabel=measure, ylabel='1-CDF(x)',
                                   marker='o')
            if m is not None:
                lambdas.append(m)
            fig1.show()
            dir_name = './results/'+city+'/centrality_measures/distributions/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig_name = dir_name+'distr_plot_'+measure+'_centrality.png'
            fig1.savefig(fig_name)

            ''' Plot network with 20% of significant nodes with color depending on value of centrality '''
            fig2 = nf.plot_network_with_node_color_based_on_measure(net=net, measures=values,
                                                                    title=measure+' centrality')
            fig2.show()
            dir_name = './results/'+city+'/centrality_measures/network/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig_name = dir_name+'network_'+measure+'_centrality.png'
            fig2.savefig(fig_name)
            plt.close('all')

    ''' Create table for betweenness analysis '''

    pd.options.display.float_format = lambda x: '{:,.3f}'.format(x)
    pd.set_option('colheader_justify', 'center')
    html_string = nf.get_html_string()

    df_bc_k = pd.DataFrame.from_dict(degree_betweenness).describe()
    df = pd.DataFrame.from_dict(nf.load_json(centralities_path)).T
    df.fillna(0.0)

    df_bc_cols = ['City', 'max(g(i))', '<g(i)>', 'n', 'max(g(k))', '<g(k)>', 'y']  # \u03B7
    df_bc = pd.DataFrame(columns=df_bc_cols)
    df_bc['City'] = cities
    df_bc['max(g(i))'] = [np.max(x) for x in df.betweenness]
    df_bc['<g(i)>'] = [np.mean(x) for x in df.betweenness]
    df_bc['lambda'] = lambdas
    df_bc['max(g(k))'] = np.array(df_bc_k.loc['max'])
    df_bc['<g(k)>'] = np.array(df_bc_k.loc['mean'])
    df_bc['eta'] = etas

    print(df_bc)

    df_bc.to_csv('./results/all/tables/betweenness.csv')

    # with open('./results/all/tables/betweenness_html.html', 'w') as f:
    #     f.write(html_string.format(table=df_bc.to_html(classes='mystyle')))

    ''' Plot all distributions of the same centrality type for all the cities coloring with colormap the
    # line depending on the area of the city and the population of it  '''
    for measure, values in centrality_dict.items():

        fig = nf.plot_multiple_ccdf_with_colorbar(datavecs=values, labels=cities, xlabel=measure, ylabel='1-CDF',
                                                  c=areas)
        # fig.show()
        fig_name = './results/all/plots/centrality_measures/ccdf_'+measure+'_centrality_area.png'
        fig.savefig(fig_name, bbox_inches='tight')

        fig = nf.plot_multiple_ccdf_with_colorbar(datavecs=values, labels=cities, xlabel=measure, ylabel='1-CDF',
                                                  c=populations)
        # fig.show()
        fig_name = './results/all/plots/centrality_measures/ccdf_' + measure + '_centrality_population.png'
        fig.savefig(fig_name, bbox_inches='tight')

    ''' Plot overall correlation degree-avg betweenness '''
    x_values, y_values = [], []
    for city in cities:
        x_values.append(list(degree_betweenness[city].keys()))
        y_values.append(list(degree_betweenness[city].values()))

    fig = nf.plot_multiple_distributions_with_colorbar_log_log(x_values=x_values, y_values=y_values, labels=cities,
                                                               xlabel='k', ylabel='<g(k)>', c=areas)
    fig_name = './results/all/plots/centrality_measures/k_avg_betweenness.png'
    fig.savefig(fig_name, bbox_inches='tight')
