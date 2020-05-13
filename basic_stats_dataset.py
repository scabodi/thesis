import network_functions as nf
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    cities = nf.get_list_cities_names()

    types = nf.get_types_of_transport_and_colors()
    labels = {i: name for i, name in enumerate(nf.get_types_of_transport_names())}
    count_types = {i: 0 for i in range(8)}

    types_for_cities = {city: {type: 0 for type in labels.values()} for city in cities}

    for city in cities:

        path = 'data/' + city + '/network_combined.csv'
        t = set(nf.get_types_for_city(path))
        # types_for_cities[city] = t
        for type in t:
            types_for_cities[city][labels[type]] = 1
            count_types[type] += 1

    df = pd.DataFrame.from_dict(types_for_cities).T

    ''' Dataframe with types of transport and percentages of presence in cities'''
    df1 = round(df.sum(axis=0)/27, 2)

    df1.plot(kind='bar', rot=70, alpha=0.5, title='% types of transportation in dataset')
    fig_name = 'results/all/plots/stats/perc_types_of_transport.png'
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()

    ''' Dataframe with cities and how many different types of transport they have '''
    df2 = df.sum(axis=1)
    df2.plot(kind='bar', rot=70, alpha=0.5, title='Number of types of transport for city')
    fig_name = 'results/all/plots/stats/num_types_of_transport_per_city.png'
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
