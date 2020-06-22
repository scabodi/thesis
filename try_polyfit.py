import network_functions as nf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    degrees = nf.load_json('./results/all/json/degrees.json')
    area_population_dict = nf.load_json('./results/all/json/area_population.json')
    d = nf.order_dict_based_on_list_keys(area_population_dict, sorted(area_population_dict.keys()))
    df_area_population = pd.DataFrame.from_dict(d).T
    areas = df_area_population['area']
    prefix_png = './results/all/plots/'

    figs, fig_names = [], []

    ''' Plot degree distributions for all cities '''
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
    fig.show()
    fig_name = 'results/all/plots/basic_measures/degree_distribution.png'
    # fig.savefig(fig_name, bbox_inches='tight')
    # figs.append(fig)
    # fig_names.append(fig_name)
    # plt.clf()

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
        fig.show()
        fig_name = prefix_png+'basic_measures/clusters/'+cluster+'_P(k).png'
        # fig.savefig(fig_name, bbox_inches='tight')
        figs.append(fig)
        fig_names.append(fig_name)
        plt.clf()

    ''' Plot the degree distribution for each city and save the m of the fit line '''
    gammas = []
    for city, list_degrees in degrees.items():
        plt.close('all')
        fig, m = nf.plot_distribution_log_log(datavec=list_degrees, label=city, xlabel='k', ylabel='P(k)',
                                              max_x=int(df_degrees[city][1]))
        fig_name = 'results/'+city+'/degree_distribution.png'
        # fig.savefig(fig_name, bbox_inches='tight')
        figs.append(fig)
        fig_names.append(fig_name)
        plt.clf()
        gammas.append(m)

    df_degrees = df_degrees.T
    df_degrees['\u03B3'] = gammas
    # nf.dump_json('./results/all/json/gammas.txt', gammas)
    # print(df_degrees)
    # print(df_degrees.describe())

    for fig, fig_name in zip(figs, fig_names):
        fig.savefig(fig_name,  bbox_inches='tight')
