import time
import network_functions as nf
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os


def plot_network_with_nodes_colormap(net, measure_name, max, title, type):
    """
    Plot city network with nodes colored depending on centrality measure (NOTE: not all nodes!)

    :param net: networkx obj
    :param measure_name: str - type of centrality measure

    :return: fig
    """

    fig = plt.figure(figsize=(12, 12))

    measures = [net.node[n][measure_name] for n in net.nodes()]
    # top_20_percent = measures
    # if type == 3:
    #     top_20_percent = nf.get_ordered_x_percent(measures, 0.5)

    sub_nodes, top = [], []
    for node, measure in zip(net.nodes(), measures):
        if measure > 0:
            top.append(measure)
            sub_nodes.append(node)

    if not sub_nodes:
        return None
    nodes = nx.draw_networkx_nodes(net, pos=nx.get_node_attributes(net, 'pos'), node_size=50, cmap=plt.cm.plasma,
                                   node_color=top, alpha=0.8, vmax=max,
                                   nodelist=sub_nodes)

    nx.draw_networkx_edges(net, pos=nx.get_node_attributes(net, 'pos'), alpha=0.2)

    plt.title(title)
    fig.colorbar(nodes)
    plt.axis('off')
    return fig


def set_net_attributes_and_plot(net, attr_name, attr_dict, type, dir_plots, dict_number_types):

    # add attributes to nodes in the network
    nx.set_node_attributes(net, attr_name, attr_dict[type])
    max_peak = max(attr_dict[type].values())

    # plot network with colors based on the value of the attribute
    title = city + ' '+attr_name+' ' + dict_number_types[type] + ' transport network'
    fig = plot_network_with_nodes_colormap(net, attr_name, max_peak, title, type)
    if fig is not None:
        fig_name = dir_plots + dict_number_types[type] + '_'+attr_name+'.png'
        fig.savefig(fig_name, bbox_inches='tight')


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types_and_colors = nf.get_types_of_transport_and_colors()
    dict_number_types = nf.get_dict_number_type_of_transport()
    time_zones = nf.get_dict_cities_time_zones()

    json_file = './results/all/json/peak_mean_info.json'
    peak_mean_info = nf.load_json(json_file)

    for city in cities:

        print('Processing '+city+' ...')

        data_file = './data/'+city+'/network_temporal_day.csv'
        dir_plots = './results/' + city + '/frequency_analysis/plots/'
        if not os.path.exists(dir_plots):
            os.makedirs(dir_plots)

        route_types = nf.get_types_for_city(data_file)

        net = nf.create_network(city, types=types_and_colors)
        nodes = net.nodes()
        # create two dictionaries {key = type of transport, value = {key=node, value=counter}
        # the counter is incremented when the start node is in that hour
        peak_hour_dict = {type: {node: 0 for node in nodes} for type in route_types}
        mean_hour_dict = {type: {node: 0 for node in nodes} for type in route_types}

        daily_info = pd.read_csv(data_file, delimiter=";")
        df = pd.DataFrame(daily_info, columns=['from_stop_I', 'to_stop_I', 'dep_time_ut', 'arr_time_ut', 'route_type',
                                               'trip_I', 'route_I'])

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
                    peak_hour_dict[int(type)][start] += 1
                else:
                    mean_hour_dict[int(type)][start] += 1

        for type in route_types:
            # add attributes to nodes in the network
            nx.set_node_attributes(net, 'peak_hour', peak_hour_dict[type])
            max_peak = max(peak_hour_dict[type].values())
            # plot network with colors based on the value of the attribute
            title = city+' peak hour ' + dict_number_types[type] + ' transport network'
            fig = plot_network_with_nodes_colormap(net, 'peak_hour', max_peak, title, type)
            if fig is not None:
                fig_name = dir_plots + dict_number_types[type] + '_peak_hour.png'
                fig.savefig(fig_name, bbox_inches='tight')
                if os.path.exists('./results/' + city + '/' + dict_number_types[type] + '_peak_hour.png'):
                    os.remove('./results/' + city + '/' + dict_number_types[type] + '_peak_hour.png')
                # fig.show()
                # plt.close()

            # do the same thing for the mean hour values
            nx.set_node_attributes(net, 'mean_hour',  mean_hour_dict[type])
            title = city + ' average hour ' + dict_number_types[type] + ' transport network'
            fig = plot_network_with_nodes_colormap(net, 'mean_hour', max_peak, title, type)
            if fig is not None:
                fig_name = dir_plots + dict_number_types[type] + '_mean_hour.png'
                fig.savefig(fig_name, bbox_inches='tight')
                if os.path.exists('./results/' + city + '/' + dict_number_types[type] + '_mean_hour.png'):
                    os.remove('./results/' + city + '/' + dict_number_types[type] + '_mean_hour.png')
                # fig.show()
            plt.close('all')
