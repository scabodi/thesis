import network_functions as nf
import statistics as st

if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    dict_number_types = nf.get_dict_number_type_of_transport()

    area_population_file = 'results/all/json/area_population.json'
    area_population_dict = nf.load_json(area_population_file)

    means, stdev = {}, {}

    for city in cities:
        # load info about frequencies
        json_path = './results/' + city + '/frequency_analysis/json/frequencies_info.json'
        frequencies_info = nf.load_json(json_path)

        # store info about mean and standard deviation for each type of transport and each city
        for type_of_transport, dict_hour_freq in frequencies_info.items():
            type = int(type_of_transport)
            if type not in means and type not in stdev:
                means[type], stdev[type] = {}, {}
            values = [x for x in dict_hour_freq.values() if x > 0]
            # values = dict_hour_freq.values()
            means[type][city] = st.mean(values)
            stdev[type][city] = st.stdev(values)

    ''' Order all dictionaries based on means of frequency for each city'''
    ordered_means, ordered_stdev, area_population_dict_ordered = {}, {}, {}
    for type_of_transport in means.keys():
        ordered_means[type_of_transport] = nf.order_dict_based_on_values(means[type_of_transport])
        ordered_stdev[type_of_transport] = nf.order_dict_based_on_list_keys(stdev[type_of_transport],
                                                                            ordered_means[type_of_transport].keys())

    ''' Plot bars for mean and standard deviations with population info - all cities divided by type of transport '''
    for type in means.keys():

        labels = ordered_means[type].keys()
        area_population_dict_ordered = nf.order_dict_based_on_list_keys(area_population_dict, labels, True)
        populations = [v['population'] for k, v in area_population_dict_ordered.items()
                       if k in ordered_means[type].keys()]

        fig = nf.plot_bars_mu_st(labels=labels, mus=ordered_means[type].values(),
                                 sts=ordered_stdev[type].values(), ylabel='Number of vehicles',
                                 type=type, feature=populations, feature_label='Population')
        # fig.show()
        fig_name = './results/all/plots/frequencies/'+dict_number_types[type]+'_bar_mu_st.png'
        fig.savefig(fig_name, bbox_inches='tight')
