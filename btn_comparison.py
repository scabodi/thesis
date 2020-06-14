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

    df = pd.read_csv('./results/all/tables/frequencies.csv')
    df = df.rename(columns={'# lines': 'L', '# vehicles PH': '#ph', '# vehicles MH': '#mh', 'Total # vehicles': 'tot#',
                            'Peak hour': 'ph', 'Mean hour': 'mh'})
    df['Continent'] = nf.get_city_continent_dict().values()

    df_ = df.sort_values(by=['#ph'])
    # print(df1)
    # print(df.describe())
    # print(df['mh'].value_counts())
    #
    print(df.loc[df['Continent'] == 'Oceania'])
    # print(df.loc[df['# lines'] <= 420])

    df1 = pd.read_csv('./results/all/tables/measures.csv')
    df1.set_index("City", inplace=True)
    # print(df1)

    # df2 = pd.concat([df, df1], axis=1, join='inner')
    df2 = pd.merge(df, df1, on='City', how='outer')
    df2.set_index("City", inplace=True)
    # print(df2)

    df3 = df2[['L', '#ph', '#mh', 'tot#', 'A', 'P', 'N']].copy()
    # print(df3)

    # df2.plot(kind='scatter', x='# vehicles MH', y='P')
    # plt.close('all')
    # sns.regplot(df2['# vehicles PH'], df2['P'])
    # plt.show()

    print(df3.corr())
