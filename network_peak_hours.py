import time
import network_functions as nf
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types_and_colors = nf.get_types_of_transport_and_colors()
    dict_number_types = nf.get_dict_number_type_of_transport()
    time_zones = nf.get_dict_cities_time_zones()

    json_file = './results/all/json/peak_mean_info.json'
    peak_mean_info = nf.load_json(json_file)

    dump = False

    for city in cities:

        print('Processing '+city+' ...')

        data_file = './data/'+city+'/network_temporal_day.csv'
        dir_name = './results/' + city + '/frequency_analysis/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        route_types = nf.get_types_for_city(data_file)

        net = nf.create_network(city, types=types_and_colors)
        nodes = net.nodes()

        if dump:
            # create two dictionaries {key = type of transport, value = {key=node, value=counter}
            # the counter is incremented when the start node is in that hour
            peak_hour_dict = {str(type): {node: 0 for node in nodes} for type in route_types}
            mean_hour_dict = {str(type): {node: 0 for node in nodes} for type in route_types}

            daily_info = pd.read_csv(data_file, delimiter=";")
            df = pd.DataFrame(daily_info, columns=['from_stop_I', 'to_stop_I', 'dep_time_ut', 'arr_time_ut',
                                                   'route_type', 'trip_I', 'route_I'])

            time_zone = time_zones[city]*3600  # seconds to add or subtract from unix time
            for index, row in df.iterrows():
                start = row['from_stop_I']
                end = row['to_stop_I']
                unix_time = int(row['dep_time_ut'])+time_zone
                type = str(row['route_type'])
                trip_id = str(row['trip_I'])

                timestamp = ' '.join(time.ctime(unix_time).split())
                # print(timestamp)
                hour = int(timestamp.split(' ')[3][:2])
                if hour in peak_mean_info[type][city]:
                    # increase correct counter
                    index = peak_mean_info[type][city].index(hour)
                    if index == 0:
                        # peak hour
                        peak_hour_dict[type][start] += 1
                    else:
                        mean_hour_dict[type][start] += 1

            nf.dump_json(dir_name+'json/peak_hour.json', peak_hour_dict)
            nf.dump_json(dir_name + 'json/mean_hour.json', mean_hour_dict)

        else:
            peak_hour_dict = nf.load_json(dir_name+'json/peak_hour.json')
            mean_hour_dict = nf.load_json(dir_name + 'json/mean_hour.json')

        for type in route_types:

            type_of_transport = dict_number_types[type]

            dir_type = dir_name + 'plots/' + type_of_transport + '/'
            if not os.path.exists(dir_type):
                os.makedirs(dir_type)

            max_peak = max(peak_hour_dict[str(type)].values())
            # set attributes and plot network with colormap depending on number of vehicles in PEAK HOUR
            nf.set_net_attributes_and_plot(net=net, city=city, attr_name='peak_hour',
                                           attr_dict={int(k): v for k, v in peak_hour_dict[str(type)].items()},
                                           dir_plots=dir_type, type_of_transport=type_of_transport, max_peak=max_peak)

            # do the same for MEAN HOUR
            nf.set_net_attributes_and_plot(net=net, city=city, attr_name='mean_hour',
                                           attr_dict={int(k): v for k, v in mean_hour_dict[str(type)].items()},
                                           dir_plots=dir_type, type_of_transport=type_of_transport, max_peak=max_peak)
            plt.close('all')
