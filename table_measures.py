import network_functions as nf
import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc


def plot_two_colums_dataframe(df, col_x, col_y1, col_y2):

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    x, y1, y2 = df[col_x], df[col_y1], df[col_y2]

    # ax.plot(x, y1, 'bo')
    # ax2.plot(x, y2, 'r+')

    df.plot(x=col_x, y=col_y1, ax=ax, kind='scatter')
    df.plot(x=col_x, y=col_y2, ax=ax2, c='r', kind='scatter', marker='+')

    # ax = df1.plot.scatter(x='City', y='<k>', c='A', colormap='plasma')
    # ax = df1.plot.scatter(x='City', y='<k>')

    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=70)

    return fig


if __name__ == '__main__':

    cities = nf.get_list_cities_names()

    prefix_json = './results/all/json/'

    measures_file = prefix_json+'measures.json'
    area_population_file = prefix_json+'area_population.json'
    degrees_file = prefix_json+'degrees.json'
    additional_measures = prefix_json+'additional_measures.json'

    csv_path = 'results/all/tables/measures.csv'

    save = False

    ''' 
    Create a pandas dataframe to collect all the measures that I want to put in the table
    Each city is one row and the columns are the following:
        - N = #nodes
        - E = #edges
        - A = area in km^2
        - P = population 
        - <k> = avg degree
        - D = density
        - d = diameter
        - <c> = avg clustering coefficient
        - <r> = avg assortativity
        - <l> = avg path length 
    '''

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 300

    if path.exists(csv_path) and save is False:
        pd.options.display.float_format = '{:,.2f}'.format
        df = pd.read_csv(csv_path)

        df['ln(N)'] = [np.log(x) for x in df['N']]
        print(df.describe())

        # df.plot(x='N', y='P', kind='scatter')
        # plt.show()

        corr = sc.pearsonr(df['N'], df['P'])
        print(corr)
        ''' Plot the <k> for each city '''
        # df1 = df.copy().sort_values(by=['A'])
        #
        # fig = plot_two_colums_dataframe(df1, 'City', '<k>', 'A')
        # fig_name = 'results/all/plots/stats/avg_degree_vs_area.png'
        # fig.savefig(fig_name, bbox_inches='tight')
        # # plt.show()
        #
        # ''' Plot the <l> vs ln(N) in order to see if they are proportional -- plot with line fitted '''
        # plt.close()
        # sns_fig = sns.regplot(x='<l>', y='ln(N)', data=df)
        # fig = sns_fig.get_figure()
        # fig_name = 'results/all/plots/stats/l_vs_lnN.png'
        # fig.savefig(fig_name)
        # # plt.show()
        ''' Parameters of fitted line '''
        [m, q] = np.polyfit(df['<l>'], df['ln(N)'], 1)

        print(m)
        print(df[df['<k>'] == df['<k>'].min()])
        # df.describe().to_csv('results/all/tables/describe.csv', index=False, float_format='%.2f')

    else:

        mdf = pd.read_json(measures_file).T
        ap_df = pd.read_json(area_population_file).T
        m = nf.load_json(additional_measures)

        cols = ['City', 'N', 'E', 'A', 'P', '<k>', 'd', '<c>', '<r>', '<l>']
        df = pd.DataFrame(columns=cols)
        df['City'] = cities
        df['N'] = [int(x) for x in mdf['#nodes']]
        df['E'] = [int(x) for x in mdf['#edges']]
        # df['D'] = [x for x in mdf['density']]
        df['d'] = [x for x in mdf['diameter']]
        df['<c>'] = [round(x, 3) for x in mdf['avg_cc']]
        df['A'] = [x for x in ap_df['area']]
        df['P'] = [int(x/1000) for x in ap_df['population']]
        df['<k>'] = round(2*df['E']/df['N'], 2)
        df['<r>'] = [round(x, 2) for x in m['<r>']]
        df['<l>'] = [round(x, 2) for x in m['<l>']]

        print(df)

        df.to_csv(csv_path, index=False)
