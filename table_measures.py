import network_functions as nf
import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


if __name__ == '__main__':

    cities = nf.get_list_cities_names()

    prefix_json = './results/all/json/'

    measures_file = prefix_json + 'measures.json'
    area_population_file = prefix_json + 'area_population.json'
    degrees_file = prefix_json + 'degrees.json'
    additional_measures = prefix_json + 'additional_measures.json'
    gammas_file = prefix_json + 'gammas.txt'

    csv_path = 'results/all/tables/measures.csv'

    save, save_html = False, False
    compare_clustering_coeff = False

    ''' 
    Create a pandas dataframe to collect all the measures that I want to put in the table
    Each city is one row and the columns are the following:
        - N = #nodes
        - E = #edges
        - A = area in km^2
        - P = population 
        - d = diameter
        - <c> = avg clustering coefficient
        - rho = N/A, density of nodes per area
        - <k> = avg degree
        - r = assortativity
        - <l> = avg path length 
        - <knn> = avg nearest neighbor
        - ln(N) = natural log of nodes
        - gamma = fitting parameter of degree distribution
        
    '''

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 300

    if compare_clustering_coeff is True:
        df_measures = pd.read_json(measures_file).T
        c_er = []
        for city in cities:
            p = float(df_measures['density'][city])
            n = int(df_measures['#nodes'][city])

            net_er = nx.fast_gnp_random_graph(n, p)
            c = nx.average_clustering(net_er)
            c_er.append(c)
        df_measures['c_er'] = [x if x > 0 else 0.000001 for x in c_er]
        df_measures['c/c_er'] = df_measures['avg_cc'] / df_measures['c_er']
        print(df_measures.describe())

    if path.exists(csv_path) and save is False:
        pd.options.display.float_format = '{:,.2f}'.format
        df = pd.read_csv(csv_path)

        ''' Plot the <k> for each city '''
        df1 = df.sort_values(by=['A'])
        fig, _ = nf.plot_two_columns_dataframe(df=df1, col_x='City', col_y1='<k>', col_y2='A')
        fig_name = 'results/all/plots/stats/avg_degree_vs_area.png'
        fig.savefig(fig_name, bbox_inches='tight')

        ''' Plot the <l> vs ln(N) in order to see if they are proportional -- plot with line fitted '''
        plt.close()
        sns_fig = sns.regplot(x='<l>', y='ln(N)', data=df)
        fig = sns_fig.get_figure()
        fig_name = 'results/all/plots/stats/l_vs_lnN.png'
        fig.savefig(fig_name)

        ''' Parameters of fitted line '''
        [m, q] = np.polyfit(df['<l>'], df['ln(N)'], 1)
        print(m)

    else:

        pd.options.display.float_format = lambda x: '{:,.0f}'.format(x) if round(x, 0) == x else '{:,.2f}'.format(x)
        pd.set_option('colheader_justify', 'center')
        html_string = nf.get_html_string()

        df_measures = pd.read_json(measures_file)
        df_area_population = pd.read_json(area_population_file)
        dict_additional = nf.load_json(additional_measures)
        gammas = nf.load_json(gammas_file)

        df = pd.concat([df_measures, df_area_population]).T
        df = df.drop('density', axis=1)
        df = df.rename(columns={'#nodes': 'N', '#edges': 'E', 'diameter': 'd', 'avg_cc': '<c>',
                                'population': 'P', 'area': 'A'})
        df['City'] = cities
        df = df[['City', 'N', 'E', 'A', 'P', 'd', '<c>']]
        df['ds'] = df['N'] / df['A']
        df['<k>'] = round(2 * df['E'] / df['N'], 2)
        df['r'] = [round(x, 2) for x in dict_additional['<r>']]
        df['<l>'] = [round(x, 2) for x in dict_additional['<l>']]
        df['<knn>'] = [np.mean(list(x.values())) for x in dict_additional['<knn>']]
        df['ln(N)'] = [np.log(x) for x in df['N']]
        df['y'] = gammas

        if save_html is True:
            df['A'] = pd.Series(["{0:,.1f}".format(val) for val in df['A']], index=df.index)
            df['<l>'] = pd.Series(["{0:,.1f}".format(val) for val in df['<l>']], index=df.index)

            with open('./results/all/tables/measures_html.html', 'w') as f:
                f.write(html_string.format(table=df.to_html(classes='mystyle')))

        print(df)
        df.to_csv(csv_path, index=False)
