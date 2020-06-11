import time
import network_functions as nf
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types_and_colors = nf.get_types_of_transport_and_colors()
    dict_number_types = nf.get_dict_number_type_of_transport()
    time_zones = nf.get_dict_cities_time_zones()

    peak_mean_info = {}

    dump = True

    pd.set_option('colheader_justify', 'center')
    # dataframe for info about all BTNs
    df_btn = pd.DataFrame(columns=['City', '# lines', 'Peak hour', '# vehicles PH', 'Mean hour', '# vehicles MH',
                                   'Total # vehicles'])
    df_btn['City'] = cities
    df_btn.set_index("City", inplace=True)

    for city in cities:

        print('Processing '+city+' ...')

        data_file = './data/'+city+'/network_temporal_day.csv'
        dir_plots = './results/'+city+'/frequency_analysis/plots/'
        if not os.path.exists(dir_plots):
            os.makedirs(dir_plots)

        ''' for each city create a dictionary of the frequencies of vehicles divided by type of transport 
            key = type of transport, value = {key = time slot, value = number of vehicles}'''
        route_types = nf.get_types_for_city(data_file)
        # dictionary for the different time slots and type of transportation
        frequency_dict = {type: {} for type in route_types}
        slots = [str(n).zfill(2) for n in range(24)]     # [00, 01, 02...] time slots of 1 hour each

        for i in route_types:
            frequency_dict[i] = {slot: set() for slot in slots}

        ''' Read the temporal information from network_temporal_day.csv '''
        daily_info = pd.read_csv(data_file, delimiter=";")
        df = pd.DataFrame(daily_info, columns=['from_stop_I', 'to_stop_I', 'dep_time_ut', 'arr_time_ut', 'route_type',
                                               'trip_I', 'route_I'])

        lines = set()  # for now just for bus transport network
        time_zone = time_zones[city]*3600  # seconds to add or subtract from unix time
        for index, row in df.iterrows():  # for each line of the dataframe (from csv file)
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
            for hour, set_trip_ids in frequency_dict[i].items():
                frequency_dict[i][hour] = len(set_trip_ids)
                # print(hour+" -> "+str(len(value)))
            print('Type of transport: '+dict_number_types[i])
            # print(frequency_dict[i])

            peak_hour = max(frequency_dict[i], key=frequency_dict[i].get)
            print('Peak hour is from %d:00 to %d:00' % (int(peak_hour), int(peak_hour)+1))
            # save info about peak hour of the city and the mean value
            mean = st.mean(frequency_dict[i].values())
            values = frequency_dict[i].values()
            closest_value = min(values, key=lambda list_value: abs(list_value - mean))
            # closest_key = int([k for k, v in frequency_dict[i].items() if v == closest_value])
            closest_key = str(list(frequency_dict[i].keys())[list(frequency_dict[i].values()).index(closest_value)])
            if str(i) not in peak_mean_info:
                peak_mean_info[str(i)] = {}
            peak_mean_info[str(i)][city] = [int(peak_hour), int(closest_key)]
            print('Closest key = %d' % int(closest_key))

            # ''' Plot #vehicles for each hour for each type of transport and city '''
            # title = city+' '+dict_number_types[i]+' transport network'
            # # nf.plot_bar_frequencies(frequency_dict[i], types_and_colors[i], fig_name, title)
            # nf.plot_bar_frequencies(frequency_dict[i], types_and_colors[i], title)
            # fig_name = dir_plots + dict_number_types[i] + '_frequency_plot_absolute.png'
            # # plt.show()
            # plt.savefig(fig_name)
            # plt.close()
            #
            # '''Plot of distributions of frequencies - all type of transport distributions'''
            # f_list = [x for x in frequency_dict[i].values() if x > 0]
            # fig = nf.plot_distribution_frequency(f_list, types_and_colors[i])
            # # fig.show()
            # fig_name = dir_plots + dict_number_types[i] + '_distribution_plot.png'
            # fig.savefig(fig_name, bbox_inches='tight')

            ''' General information '''
            n_lines = len(lines)
            # print('Number of lines: '+str(n_lines))
            # print('Peak hour frequency: '+str(frequency_dict[i][peak_hour]))
            tot_number_of_vehicles = sum(frequency_dict[i].values())
            print('Total number of vehicles for '+dict_number_types[i]+': '+str(tot_number_of_vehicles))

            # if the type is bus save the info in the dataframe
            if i == 3:  
                # save info in the dataframe
                df_btn.loc[[city], ['# lines']] = n_lines
                df_btn.loc[[city], ['Peak hour']] = peak_hour
                df_btn.loc[[city], ['# vehicles PH']] = frequency_dict[i][peak_hour]
                df_btn.loc[[city], ['Mean hour']] = closest_key
                df_btn.loc[[city], ['# vehicles MH']] = frequency_dict[i][closest_key]
                df_btn.loc[[city], ['Total # vehicles']] = tot_number_of_vehicles
                # print(df_btn)

            # print(frequency_dict)

        if dump:
            d = {}
            for k, v in frequency_dict.items():
                d[str(k)] = v
            dir_json = './results/'+city+'/frequency_analysis/json/'
            if not os.path.exists(dir_json):
                os.makedirs(dir_json)
            json_path = dir_json+'frequencies_info.json'
            nf.dump_json(json_path, d)

            if os.path.exists('./results/' + city + '/frequency_analysis/frequencies_info.json'):
                os.remove('./results/' + city + '/frequency_analysis/frequencies_info.json')

    if dump:
        json_path = './results/all/json/peak_mean_info.json'
        nf.dump_json(json_path, peak_mean_info)

        html_string = nf.get_html_string()
        with open('./results/all/tables/frequencies_html.html', 'w') as f:
            f.write(html_string.format(table=df_btn.to_html(classes='mystyle')))
