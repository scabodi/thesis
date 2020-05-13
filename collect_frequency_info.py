import time
import network_functions as nf
import pandas as pd
import statistics as st

cities = nf.get_list_cities_names()
types_and_colors = nf.get_types_of_transport_and_colors()
dict_number_types = nf.get_dict_number_type_of_transport()
time_zones = nf.get_dict_cities_time_zones()

peak_mean_info = {}

for city in cities:

    print('Processing '+city+' ...')
    data_file = './data/'+city+'/network_temporal_day.csv'
    prefix = './results/'+city+'/'
    # for each city create a dictionary of the frequencies of vehicles divided by type of transport

    route_types = nf.get_types_for_city(data_file)
    # dictionary for the different time slots and type of transportation
    frequency_dict = {type: {} for type in route_types}
    slots = [str(n).zfill(2) for n in range(24)]     # [00, 01, 02...] time slots of 1 hour each

    for i in route_types:
        frequency_dict[i] = {slot: set() for slot in slots}

    daily_info = pd.read_csv(data_file, delimiter=";")
    df = pd.DataFrame(daily_info, columns=['from_stop_I', 'to_stop_I', 'dep_time_ut', 'arr_time_ut', 'route_type',
                                           'trip_I', 'route_I'])

    lines = set()  # for now just for bus transport network
    time_zone = time_zones[city]*3600  # seconds to add or subtract from unix time
    for index, row in df.iterrows():
        start = row['from_stop_I']
        end = row['to_stop_I']
        unix_time = int(row['dep_time_ut'])+time_zone
        type = int(row['route_type'])
        trip_id = str(row['trip_I'])
        if type == 3:
            lines.add(row['route_I'])

        timestamp = ' '.join(time.ctime(unix_time).split())
        # print(timestamp)
        hour = timestamp.split(' ')[3][:2]
        frequency_dict[type][hour].add(trip_id)

    for i in route_types:
        for hour, value in frequency_dict[i].items():
            frequency_dict[i][hour] = len(value)
            # print(hour+" -> "+str(len(value)))
        print('Type of transport: '+dict_number_types[i])
        print(frequency_dict[i])

        peak_hour = int(max(frequency_dict[i], key=frequency_dict[i].get))
        print('Peak hour is from %d:00 to %d:00' % (peak_hour, peak_hour+1))
        # save info about peak hour of the city and the mean value
        mean = st.mean(frequency_dict[i].values())
        values = frequency_dict[i].values()
        closest_value = min(values, key=lambda list_value: abs(list_value - mean))
        # closest_key = int([k for k, v in frequency_dict[i].items() if v == closest_value])
        closest_key = int(list(frequency_dict[i].keys())[list(frequency_dict[i].values()).index(closest_value)])
        if str(i) not in peak_mean_info:
            peak_mean_info[str(i)] = {}
        peak_mean_info[str(i)][city] = [peak_hour, closest_key]
        print('Closest key = %d' % closest_key)

    #     ''' Plot #vehicles for each hour for each type of transport and city '''
    #     fig_name = prefix + dict_number_types[i] + '_frequency_plot_absolute.png'
    #     title = city+' '+dict_number_types[i]+' transport network'
    #     nf.plot_bar_frequencies(frequency_dict[i], types_and_colors[i], fig_name, title)
    #     # fig_name = prefix+dict_number_types[i]+'_frequency_plot_absolute.png'
    #     # fig.savefig(fig_name)
    #
    #     ''' Plot #vehicles/#lines (or routes) '''
    #     # compute number of lines for the current city and type of transport
    #     n_lines = len(lines)
    #     print('Number of lines: '+str(n_lines))
    #     # to_plot = {k: v/n_lines for k, v in frequency_dict[i].items()}
    #
    #     ''' Plot #vehicles/max of #vehicles among all slots '''
    #     max_num_vehicles = max(frequency_dict[i].values())
    #     print('Max number of vehicles for slot: '+str(max_num_vehicles))
    #     # to_plot = {k: v/max_num_vehicles for k, v in frequency_dict[i].items()}
    #
    #     ''' Plot #vehicle/#tot vehicles over all slots '''
    #     tot_number_of_vehicles = sum(frequency_dict[i].values())
    #     # TODO change customize for type of transport
    #     print('Total number of vehicles for bus: '+str(tot_number_of_vehicles))
    #     # to_plot = {k: v/tot_number_of_vehicles for k, v in frequency_dict[i].items()}
    #
    #     '''Plot of distributions of frequencies - all type of transport distributions TODO customize'''
    #     f_list = [x for x in frequency_dict[i].values() if x > 0]  # TODO decide if remove 0 or not
    #     # print(f_list)
    #     fig = nf.plot_distribution_frequency(f_list, types_and_colors[i])
    #     fig_name = prefix+dict_number_types[i]+'_distribution_plot.png'
    #     fig.savefig(fig_name, bbox_inches='tight')
    #
    # # print(frequency_dict)
    # d = {}
    # for k, v in frequency_dict.items():
    #     d[str(k)] = v
    # json_path = './results/'+city+'/frequencies_info.json'
    # nf.dump_json(json_path, d)

    json_path = './results/all/json/peak_mean_info.json'
    nf.dump_json(json_path, peak_mean_info)
