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

    dump = False

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

        dir_json = './results/' + city + '/frequency_analysis/json/'
        if not os.path.exists(dir_json):
            os.makedirs(dir_json)
        json_path = dir_json + 'frequencies_info.json'

        route_types = nf.get_types_for_city(data_file)
        frequency_dict, lines = {}, []

        if dump:
            ''' for each city create a dictionary of the frequencies of vehicles divided by type of transport 
                key = type of transport, value = {key = time slot, value = number of vehicles}'''
            # dictionary for the different time slots and type of transportation
            frequency_dict = {type: {} for type in route_types}
            slots = [str(n).zfill(2) for n in range(24)]     # [00, 01, 02...] time slots of 1 hour each

            for i in route_types:
                frequency_dict[i] = {slot: set() for slot in slots}

            ''' Read the temporal information from network_temporal_day.csv '''
            daily_info = pd.read_csv(data_file, delimiter=";")
            df = pd.DataFrame(daily_info, columns=['from_stop_I', 'to_stop_I', 'dep_time_ut', 'arr_time_ut', 'route_type',
                                                   'trip_I', 'route_I'])

            lines = set()  # just for bus transport network
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
                hour = timestamp.split(' ')[3][:2]
                frequency_dict[type][hour].add(trip_id)

        else:
            f = nf.load_json(json_path)
            for k, v in f.items():
                frequency_dict[int(k)] = v

        for i in route_types:

            print('  '+dict_number_types[i])

            dir_type = './results/' + city + '/frequency_analysis/plots/'+dict_number_types[i]+'/'
            if not os.path.exists(dir_type):
                os.makedirs(dir_type)

            if dump:
                ''' Set number of vehicles for type of transport and time slot '''
                for hour, set_trip_ids in frequency_dict[i].items():
                    frequency_dict[i][hour] = len(set_trip_ids)

                ''' Find and save info about peak hour and mean hour '''
                peak_hour, mean_hour = nf.get_mean_and_peak_hour(frequency_dict=frequency_dict, route_type=i)
                if str(i) not in peak_mean_info:
                    peak_mean_info[str(i)] = {}
                peak_mean_info[str(i)][city] = [int(peak_hour), int(mean_hour)]
                # print('Peak hour is from %d:00 to %d:00' % (int(peak_hour), int(peak_hour) + 1))
                # print('Closest key = %d' % int(closest_key))
            else:
                peak_mean_info = nf.load_json('./results/all/json/peak_mean_info.json')
                [peak_hour, mean_hour] = peak_mean_info[str(i)][city]

            ''' Plot #vehicles for each hour for each type of transport and city '''
            title = city+' '+dict_number_types[i]+' transport network'
            fig = nf.plot_bar_frequencies(frequency_dict[i], types_and_colors[i], title)
            fig_name = dir_type + dict_number_types[i] + '_frequency_plot.png'
            fig.savefig(fig_name)

            '''Plot of distributions of frequencies - all type of transport distributions'''
            f_list = [x for x in frequency_dict[i].values() if x > 0]
            fig = nf.plot_distribution_frequency(f_list, types_and_colors[i])
            fig_name = dir_type + dict_number_types[i] + '_distribution_plot.png'
            fig.savefig(fig_name, bbox_inches='tight')
            plt.close('all')

            # if the type is bus gather info and put them in the dataframe
            if i == 3 and dump:

                ''' General information '''
                n_lines = len(lines)
                tot_number_of_vehicles = sum(frequency_dict[i].values())

                # save info in the dataframe
                df_btn.loc[[city], ['# lines']] = n_lines
                df_btn.loc[[city], ['Peak hour']] = peak_hour
                df_btn.loc[[city], ['# vehicles PH']] = frequency_dict[i][peak_hour]
                df_btn.loc[[city], ['Mean hour']] = mean_hour
                df_btn.loc[[city], ['# vehicles MH']] = frequency_dict[i][mean_hour]
                df_btn.loc[[city], ['Total # vehicles']] = tot_number_of_vehicles
                # print(df_btn)

        if dump:
            d = {}
            for k, v in frequency_dict.items():
                d[str(k)] = v
            nf.dump_json(json_path, d)

    if dump:
        json_file = './results/all/json/peak_mean_info.json'
        nf.dump_json(json_file, peak_mean_info)

        html_string = nf.get_html_string()
        with open('./results/all/tables/frequencies_html.html', 'w') as f:
            f.write(html_string.format(table=df_btn.to_html(classes='mystyle')))
