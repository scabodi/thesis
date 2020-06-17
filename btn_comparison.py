import network_functions as nf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 300
    pd.options.display.float_format = lambda x: '{:,.0f}'.format(x) if round(x, 0) == x else '{:,.2f}'.format(x)

    # df = pd.read_csv('./results/all/tables/frequencies.csv')
    # df = df.rename(columns={'# lines': 'L', '# vehicles PH': '#ph', '# vehicles MH': '#mh', 'Total # vehicles': 'tot#',
    #                         'Peak hour': 'ph', 'Mean hour': 'mh'})
    # df['Continent'] = nf.get_city_continent_dict().values()
    #
    # df_ = df.sort_values(by=['#ph'])
    # print(df.loc[df['Continent'] == 'Oceania'])

    df1 = pd.read_csv('./results/all/tables/measures.csv')
    df1.set_index("City", inplace=True)

    # df2 = pd.merge(df, df1, on='City', how='outer')
    # df2.set_index("City", inplace=True)
    # #
    # df3 = df2[['L', '#ph', '#mh', 'tot#', 'A', 'P', 'N']].copy()
    # print(df3.corr())

    df_bc = pd.read_csv('./results/all/tables/betweenness.csv')
    df_bc['n'] = -df_bc['n']
    df_merge = pd.merge(df_bc, df1, on='City', how='outer')

    df_sorted = df_merge.sort_values(by='y_x')
    print(df_sorted)

    # df_sorted.plot(kind='scatter', x='N', y='y_x', labels='City')
    plt.scatter(df_sorted['N'], df_sorted['y_x'])
    [m, b] = np.polyfit(df_sorted['N'], df_sorted['y_x'], 1)
    plt.plot(df_sorted['N'], m * df_sorted['N'] + b)

    plt.show()

    df_corr = df_merge[['City', 'n', 'y_x', 'N', 'A', 'P']].copy()
    print(df_corr.corr())
