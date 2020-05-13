import network_functions as nf
import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    cities = nf.get_list_cities_names()

    prefix_json = './results/all/json/'

    measures_file = prefix_json+'measures.json'
    area_population_file = prefix_json+'area_population.json'
    degrees_file = prefix_json+'degrees.json'
    additional_measures = prefix_json+'additional_measures.json'

    csv_path = 'results/all/tables/measures.csv'

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

    if path.exists(csv_path):
        pd.options.display.float_format = '{:,.2f}'.format
        df = pd.read_csv(csv_path)

        df['ln(N)'] = [np.log(x) for x in df['N']]
        print(df.describe())

        ''' Plot the <l> vs ln(N) in order to see if they are proportional -- plot with line fitted '''
        plt.close()
        sns_fig = sns.regplot(x='<l>', y='ln(N)', data=df)
        fig = sns_fig.get_figure()
        fig_name = 'results/all/plots/stats/l_vs_lnN.png'
        fig.savefig(fig_name)
        # plt.show()
        ''' Parameters of fitted line '''
        [m, q] = np.polyfit(df['<l>'], df['ln(N)'], 1)

        print(m)
        # print(df[df['<l>'] == df['<l>'].max()])
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
        df['<k>'] = round(df['E']/df['N'], 2)
        df['<r>'] = [round(x, 2) for x in m['<r>']]
        df['<l>'] = [round(x, 2) for x in m['<l>']]

        print(df)

        df.to_csv(csv_path, index=False)
