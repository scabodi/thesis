import network_functions as nf
import statistics as st


cities = nf.get_list_cities_names()
dict_number_types = nf.get_dict_number_type_of_transport()

area_population_file = 'results/all/json/area_population.json'
area_population_dict = nf.load_json(area_population_file)

means, stdev = {}, {}

for city in cities:
    # load info about frequencies
    json_path = './results/' + city + '/frequencies_info.json'
    d = nf.load_json(json_path)

    # store info about mean and standard deviation for each type of transport and each city
    for number, dict_values in d.items():
        type = int(number)
        if type not in means and type not in stdev:
            means[type], stdev[type] = {}, {}
        # values = [x for x in dict_values.values() if x > 0]
        values = dict_values.values()
        means[type][city] = st.mean(values)
        stdev[type][city] = st.stdev(values)

''' Order all dictionaries based on means of frequency for each city'''
ordered_means, ordered_stdev, area_population_dict_ordered = {}, {}, {}
for k, v in means.items():
    ordered_means[k] = nf.order_dict_based_on_values(v)
labels = ordered_means[3].keys()
for k, v in stdev.items():
    ordered_stdev[k] = nf.order_dict_based_on_list_keys(v, labels, True)
area_population_dict_ordered = nf.order_dict_based_on_list_keys(area_population_dict, labels)


''' Plot bars for mean and standard deviations with population info - all cities divided by type of transport '''
for type in means.keys():

    populations = [v['population'] for k, v in area_population_dict_ordered.items() if k in ordered_means[type].keys()]

    fig = nf.plot_bars_frequency_mu_st(ordered_means[type].keys(), ordered_means[type].values(),
                                       ordered_stdev[type].values(), type, populations)
    # fig.show()
    fig_name = './results/all/plots/frequencies/'+dict_number_types[type]+'_bar_mu_st.png'
    fig.savefig(fig_name, bbox_inches='tight')
