import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import json
import pandas as pd
from geopy.distance import geodesic
from collections import deque
import statistics as st
import seaborn as sns
import matplotlib.colors as mc
from colormap import rgb2hex, rgb2hls, hls2rgb
from scipy.spatial import distance
import os

''' JSON '''


def load_json(file):
    """
    Load dictionary from json file

    :param file: str - name of file to get

    :return: dict: data in dictionary format
    """
    with open(file) as json_file:
        d = json.load(json_file)
    return d


def dump_json(file, d):
    """
    Dump dictionary into a json file

    :param file: str - path to file
    :param d: dictionary to be put into the file
    :return: nothing
    """

    with open(file, 'w') as fp:
        json.dump(d, fp)


''' Get cities PARAMETERS (colors, type of transport, cities, etc.)  '''


def get_types_of_transport_and_colors():
    return {0: 'g', 1: 'orange', 2: 'r', 3: 'b', 4: 'aqua', 5: 'y', 6: 'm', 7: 'maroon'}


def get_types_of_transport_names():
    return ['tram', 'subway', 'rail', 'bus', 'ferry', 'cablecar', 'gondola', 'funicular']


def get_dict_number_type_of_transport():
    return {0: 'tram', 1: 'subway', 2: 'rail', 3: 'bus', 4: 'ferry', 5: 'cablecar', 6: 'gondola', 7: 'funicular'}


def get_list_cities_names():
    cities = ['adelaide', 'antofagasta', 'athens', 'belfast', 'berlin', 'bordeaux', 'brisbane', 'canberra',
              'detroit', 'dublin', 'grenoble', 'helsinki', 'kuopio', 'lisbon', 'luxembourg', 'melbourne',
              'nantes', 'palermo', 'paris', 'prague', 'rennes', 'rome', 'sydney', 'toulouse', 'turku',
              'venice', 'winnipeg']
    return cities


def get_dict_cities_time_zones():
    d = {
        'adelaide': 10.5, 'antofagasta': -4, 'athens': 2, 'belfast': 1, 'berlin': 2, 'bordeaux': 1, 'brisbane': 10,
        'canberra': 11, 'detroit': -5, 'dublin': 0, 'grenoble': 1, 'helsinki': 2, 'kuopio': 2, 'lisbon': 0,
        'luxembourg': 1, 'melbourne': 11, 'nantes': 1, 'palermo': 2, 'paris': 1, 'prague': 1, 'rennes': 1, 'rome': 1,
        'sydney': 11, 'toulouse': 1, 'turku': 2, 'venice': 1, 'winnipeg': -6,
    }
    return d


def get_list_sorted_values(value_str, dict):
    result = [v[value_str] for k, v in sorted(dict.items())]
    return result


def get_cluster_dict_for_area():
    result = {'0-100': ['kuopio', 'rennes', 'luxembourg', 'antofagasta'],
              '100-300': ['paris', 'palermo', 'turku', 'belfast'],
              '300-500': ['dublin', 'winnipeg', 'canberra', 'athens', 'venice', 'rome', 'prague'],
              '500-1000': ['grenoble', 'nantes', 'helsinki', 'toulouse', 'adelaide'],
              '1000+': ['berlin', 'bordeaux', 'lisbon', 'brisbane', 'sydney', 'melbourne', 'detroit']}
    return result


def get_capital_cities():
    result = ['athens', 'berlin', 'dublin', 'helsinki', 'lisbon', 'luxembourg', 'paris', 'prague', 'rome', 'sydney']
    return result


def get_capitals_with_central_station_node():
    result = {'athens': 461, 'berlin': 146, 'dublin': 332, 'helsinki': 22, 'lisbon': 1490, 'luxembourg': 1356,
              'paris': 953, 'prague': 484, 'rome': 477, 'sydney': 36611}
    return result


def centeroid_np(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def get_central_node(coords):
    # if the results have already been saved in a file, retrieve them
    central_nodes_json = 'results/all/json/central_nodes.json'
    if os.path.exists(central_nodes_json) and os.path.isfile(central_nodes_json):
        return load_json(central_nodes_json)

    # otherwise, calculate central node of the biggest connected component
    # find barycenter
    a = np.array(list(coords.values()))
    centre = centeroid_np(a)
    # look for nearest node in the network
    nearest = tuple(min(a, key=lambda x: distance.euclidean(x, centre)))
    central_node = None
    for n, coord in coords.items():
        if coord == nearest:
            central_node = n
            break
    return central_node


def get_types_for_city(path):

    info = pd.read_csv(path, delimiter=";")
    df = pd.DataFrame(info, columns=['route_type'])

    return df.route_type.unique()


def order_dict_based_on_values(d):

    ordered_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    return ordered_d


def order_dict_based_on_list_keys(d, l, condition=False):

    if condition:
        ordered_d = {k: d[k] for k in l if k in d}
    else:
        ordered_d = {k: d[k] for k in l}
    return ordered_d


def get_html_string():
    html_string = '''
                    <html>
                      <head><title>Table of network measures</title></head>
                      <link rel="stylesheet" type="text/css" href="df_style.css"/>
                      <body>
                        {table}
                      </body>
                    </html>.
                    '''
    return html_string


''' CREATE AND PLOT NETWORK '''


def load_nodes(net, path):
    """
    Load nodes from specific file and add them to the network obj

    :param net: networkx object
    :param path: path to retrieve info about nodes (stops)
    """

    nodes_info = pd.read_csv(path, delimiter=";")
    df = pd.DataFrame(nodes_info, columns=['stop_I', 'lat', 'lon', 'name'])
    # print(df)
    for index, row in df.iterrows():
        net.add_node(row['stop_I'], coords=(row['lat'], row['lon']), pos=(row['lon'], row['lat']))


def load_edges(net, path, types):
    """
    Load edges from specific file and add them to the network obj

    :param types: dict of types of route with each color assigned
    :param net: networkx object
    :param path: path to retrieve info about edges

    """

    edges_info = pd.read_csv(path, delimiter=";")
    df = pd.DataFrame(edges_info, columns=['from_stop_I', 'to_stop_I', 'd', 'duration_avg', 'n_vehicles',
                                           'route_I_counts', 'route_type'])
    for index, row in df.iterrows():
        net.add_edge(row['from_stop_I'], row['to_stop_I'],  color=types[row['route_type']])


def create_network(city, types=None, edges_file=None):
    """
    Create network from g (if g is not None) or from cvs files

    :param types: dict of types of route with each color assigned
    :param city: str - city name
    :param edges_file: str - path to file for edges - used for different network type (bus, tram etc.)

    :return: net: networkx object
    """
    nodes_path = 'data/'+city+'/network_nodes.csv'
    edges_path = 'data/' + city + '/network_combined.csv'
    if edges_file is not None:
        edges_path = edges_file

    net = nx.Graph()
    load_nodes(net, nodes_path)
    load_edges(net, edges_path, types)

    return net


def make_proxy(clr, **kwargs):
    return Line2D([0, 1], [0, 1], color=clr, **kwargs)


def plot_network(net, node_list=None):
    """
    Plot networkx object for the specified city
    Each edge has its color depending on the route type and the node shape is '' because they are not displayed

    :param net: networkx obj
    :param node_list: nodes to be represented
    """
    colors = nx.get_edge_attributes(net, 'color').values()
    all_colors = set(colors)
    all_edge_types = {'g': 'Tram', 'orange': 'Subway', 'r': 'Rail', 'b': 'Bus', 'aqua': 'Ferry',
                      'y': 'Cable car', 'm': 'Gondola', 'maroon': 'Funicular'}
    edge_types = {color: all_edge_types[color] for color in all_colors}

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if node_list is None:
        nx.draw_networkx(net, ax=ax, pos=nx.get_node_attributes(net, 'pos'), with_labels=False, node_size=5,
                         node_shape='', edge_color=colors, alpha=0.5)
        proxies = [make_proxy(clr, lw=5) for clr in edge_types.keys()]
        labels = [edge_type for clr, edge_type in edge_types.items()]
        plt.legend(proxies, labels)
    else:
        nx.draw_networkx(net, ax=ax, pos=nx.get_node_attributes(net, 'pos'), with_labels=False, node_size=100,
                         nodelist=node_list, node_color='red', alpha=0.5, edge_color='gray')

    # to change background color --> ax.set_facecolor('k')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()

    # fig_name = './data/' + city + '/network.png'
    # fig.savefig(fig_name)
    # plt.close()
    return fig, ax


''' Compute NETWORK MEASURES (BASIC, ADDITIONAL and CENTRALITY) '''


def compute_measures(net, dict):
    """
    Compute some network measures and add them to a dictionary

    :param net: networkx obj
    :param dict: dictionary where to add info about computed measures

    """
    # N -- number of network nodes
    N = nx.number_of_nodes(net)
    dict['#nodes'] = N

    # L -- number of links
    L = nx.number_of_edges(net)
    dict['#edges'] = L

    # D -- density
    D = nx.density(net)
    dict['density'] = D

    # d -- diameter
    max_sub = max(nx.connected_component_subgraphs(net), key=len)
    d = nx.diameter(max_sub)
    dict['diameter'] = d

    # C -- average clustering coefficient
    C = nx.average_clustering(net, count_zeros=True)
    dict['avg_cc'] = C


def get_assortativity(net):
    return nx.degree_pearson_correlation_coefficient(net)


def get_avg_shortest_path_length(net):
    max_component = max(nx.connected_component_subgraphs(net), key=len)
    return nx.average_shortest_path_length(max_component)


def get_avg_degree_connectivity(net):
    return nx.average_degree_connectivity(net)


def get_centrality_measures(network, tol):
    """
    Calculates five centrality measures (degree, betweenness, closeness, and
    eigenvector centrality, and k-shell) for the nodes of the given network.

    :param network: networkx.Graph()
    :param tol: tolerance parameter for calculating eigenvector centrality

    :return [degree, betweenness, closeness, eigenvector_centrality]: list of numpy.arrays
    """

    nodes = network.nodes()

    degree_centr = nx.degree_centrality(network)
    degree_centr = sorted(degree_centr.items(), key=lambda pair: list(nodes).index(pair[0]))
    degree = np.array([v for k, v in degree_centr])

    betweenness_centr = nx.betweenness_centrality(network, normalized=True)
    betweenness_centr = sorted(betweenness_centr.items(), key=lambda pair: list(nodes).index(pair[0]))
    betweenness = np.array([v for k, v in betweenness_centr])

    closeness_centr = nx.closeness_centrality(network)
    closeness_centr = sorted(closeness_centr.items(), key=lambda pair: list(nodes).index(pair[0]))
    closeness = np.array([v for k, v in closeness_centr])

    eigenvector_centr = nx.eigenvector_centrality(network, tol=tol)
    eigenvector_centr = sorted(eigenvector_centr.items(), key=lambda pair: list(nodes).index(pair[0]))
    eigenvector_centrality = np.array([v for k, v in eigenvector_centr])

    return [degree, betweenness, closeness, eigenvector_centrality]


def set_degree_betweenness_dict_for_city(net, dict_k_b, city):
    betweenness = nx.betweenness_centrality(net)
    k_b = {}
    for node in net.nodes():
        k = net.degree(node)
        if k not in k_b:
            k_b[k] = []
        k_b[k].append(betweenness[node])

    for k in sorted(k_b.keys()):
        avg = np.mean(k_b[k])
        dict_k_b[city][k] = avg


''' Plot NETWORK with nodes colorbar depending on some measures '''


def get_ordered_x_percent(measures, percentage):
    """
    :param measures: list of int or float
    :param percentage: float to be multiplied (20% -> 0.2)

    :return: ordered_measures: list of first 20% of ordered input list
    """

    ordered_measures = measures.copy()
    ordered_measures.sort(reverse=True)
    ordered_measures = ordered_measures[: int(len(measures) * percentage)]

    return ordered_measures


def get_normalized_values(values):

    minimum = min(values)
    maximum = max(values)
    res = []
    for v in values:
        z = (v-minimum)/(maximum-minimum)
        res.append(z)

    return res


def plot_network_with_node_color_based_on_measure(net, measures, title):
    """
    Plot city network with nodes colored depending on centrality measure (NOTE: not all nodes!)

    :param net: networkx obj
    :param measures: list of floats
    :param title: str - title of the plot

    :return: fig
    """

    fig = plt.figure(figsize=(12, 12))

    top_20_percent = get_ordered_x_percent(measures, 0.2)

    sub_nodes, top = [], []
    for node, measure in zip(net.nodes(), measures):
        if measure in top_20_percent:
            top.append(measure)
            sub_nodes.append(node)

    nodes = nx.draw_networkx_nodes(net, pos=nx.get_node_attributes(net, 'pos'), node_size=[v*10**5 for v in top],
                                   cmap=plt.cm.plasma, node_color=top, alpha=0.8, nodelist=sub_nodes)
    nx.draw_networkx_edges(net, pos=nx.get_node_attributes(net, 'pos'), alpha=0.2)

    plt.title(title)
    fig.colorbar(nodes)
    plt.axis('off')
    return fig


''' Plot DISTRIBUTIONS -- TO REVIEW '''


def plot_distribution(x_values, y_values, x_label, y_label):
    """
    Plot distribution of x and y values

    :param x_values: list of x values (int or float)
    :param y_values: list of y values (int or float)
    :param x_label: str
    :param y_label: str

    :return: fig
    """

    fig = plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return fig


def plot_ccdf(datavecs, labels, xlabel, ylabel, marker=None):

    """
    Plots in a single figure the complementary cumulative distributions (1-CDFs)
    of the given data vectors.

    :param datavecs: data vectors to plot, a list of iterables
    :param labels: labels for the data vectors, list of strings
    :param xlabel: x label for the figure, string
    :param ylabel: y label for the figure, string
    :param marker: marker for the line

    :return: fig
    """

    styles = ['-', '--', '-.', ':']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = None
    for datavec, label, style in zip(datavecs, labels, styles):
        sorted_vals = np.sort(np.unique(datavec))
        ccdf = np.zeros(len(sorted_vals))
        n = float(len(datavec))
        for i, val in enumerate(sorted_vals):
            ccdf[i] = np.sum(datavec >= val) / n
        # x_values = range(1, len(sorted_vals) + 1)
        if marker is not None:
            ax.loglog(sorted_vals, ccdf, linestyle=' ', label=label, marker=marker)
        else:
            ax.loglog(sorted_vals, ccdf, linestyle=style, label=label)
        ''' For a less precise and faster execution use:
        sorted_vals = np.sort(datavec)
        ccdf = np.linspace(1, 1./len(datavec), len(datavec))
        ax.loglog(sorted_vals, ccdf, linestyle=style, label=label)'''
        if xlabel == 'betweenness':
            [m, q] = np.polyfit(sorted_vals, np.log(ccdf), 1)
            print("m: %.4f, q: %.4f" % (m, q))
            y_fit = np.exp(m * sorted_vals + q)
            ax.plot(sorted_vals, y_fit, linestyle=':', label='\u03B7 = %.2f' % m)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=0)
    ax.grid()
    fig.tight_layout()

    return fig, m


def plot_multiple_ccdf_with_colorbar(datavecs, labels, xlabel, ylabel, c):

    """
    Plots in a single figure the complementary cumulative distributions (1-CDFs)
    of the given data vectors.

    :param datavecs: data vectors to plot, a list of iterables
    :param labels: labels for the data vectors, list of strings
    :param xlabel: x label for the figure, string
    :param ylabel: y label for the figure, string
    :param c: list of values that gives the range of colors in the colorbar

    :return: fig
    """

    # styles = (['-', '--', '-.', ':']*7)[:len(datavecs)]
    n = len(datavecs)
    markers = (['_', 'v', '^', 'o', '+', 'x', 'd'] * 7)[:n]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.plasma(get_normalized_values(c))
    # low_limit = 10**-4

    for datavec, label, marker, color in zip(datavecs, labels, markers, colors):
        # sorted_datavec = [x for x in sorted(datavec) if x > low_limit]
        sorted_datavec = sorted(datavec)
        cdf = np.zeros(len(sorted_datavec))
        for i, val in enumerate(sorted_datavec, 0):
            cdf[i] = val
            if i > 0:
                cdf[i] += cdf[i-1]
        sum_tot = cdf[len(cdf)-1]
        ccdf = [1-(x/sum_tot) for x in cdf]
        ax.loglog(sorted_datavec, ccdf, marker=marker, label=label, linestyle=' ', color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.1)
    ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.5, .102), loc='lower left', ncol=5, mode="expand", borderaxespad=0.1)

    m = mpl.cm.ScalarMappable(cmap=mpl.cm.plasma)
    m.set_array(c)
    fig.colorbar(m)

    return fig


def plot_multiple_distributions_with_colorbar_log_log(x_values, y_values, labels, xlabel, ylabel, c):

    n = len(labels)
    markers = (['_', 'v', '^', 'o', '+', 'x', 'd'] * 7)[:n]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.plasma(get_normalized_values(c))

    for xs, ys, label, marker, color in zip(x_values, y_values, labels, markers, colors):
        ax.loglog(xs, ys, marker=marker, label=label, linestyle=' ', color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.5, .102), loc='lower left', ncol=5, mode="expand", borderaxespad=0.1)

    m = mpl.cm.ScalarMappable(cmap=mpl.cm.plasma)
    m.set_array(c)
    fig.colorbar(m)

    return fig


def plot_correlation_measures_log_log(x_values, y_values, xlabel, ylabel, title):

    fig, ax = plt.subplots()

    ax.plot(x_values, y_values, marker='o', linestyle=' ')
    ax.set_yscale('log')
    ax.set_xscale('log')

    logx = np.log(x_values[1:])
    logy = np.log(y_values[1:])
    [m, q] = np.polyfit(logx, logy, 1)
    print("m: %.4f, q: %.4f" % (m, q))
    y_fit = np.exp(m * logx + q)
    ax.plot(x_values[1:], y_fit, linestyle=':', label='\u03B7 = %.2f' % m)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    ax.legend()

    return fig, m


def plot_multiple_lines(x_values, y_values, labels, xlabel, ylabel):

    """
    Plots in a single figure the complementary cumulative distributions (1-CDFs)
    of the given data vectors.

    :param x_values: list of int or floats (in this case nodes)
    :param y_values: list of lists of floats
    :param labels: labels for the data vectors, list of strings
    :param xlabel: x label for the figure, string
    :param ylabel: y label for the figure, string

    :return: fig
    """
    styles = ['-', '--', '-.', ':']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for y_value, label, style in zip(y_values, labels, styles):
        sorted_y_value = sorted(y_value)
        ax.plot(x_values, sorted_y_value, linestyle=style, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=0)
    ax.grid(alpha=0.2)
    # plt.yscale('log')

    return fig


def plot_distribution_log_log(datavec, label, xlabel, ylabel, max_x):
    """
    Plot distribution with fit line

    :param datavec: list of int - degrees of nodes
    :param label: str
    :param xlabel: str
    :param ylabel: str
    :param max_x: int - max value

    :return: fig, m
    """

    x_values = range(1, max_x + 1)
    counter = [0] * max_x
    for value in datavec:
        counter[value - 1] += 1
    px = np.array([v / len(datavec) for v in counter])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(x_values, px, marker='o', label=label, linestyle=' ')

    y = [v for v in px[1:] if v > 0]
    x = range(2, len(y) + 2)
    logx = np.log(x)
    logy = np.log(y)
    [m, q] = np.polyfit(logx, logy, 1)
    # print("m: %.4f, q: %.4f" % (m, q))
    y_fit = np.exp(m * logx + q)
    m = -m
    ax.plot(x, y_fit, linestyle=':', label='\u03B3 = %.2f' % m)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()

    return fig, m


def plot_multiple_distributions_with_colorbar_log_log_and_fitted_line(datavecs, labels, xlabel, ylabel, c, max_x):
    """
    Function that plots several distribution over a log-log plot with a line fit over the average of all the
    distributions

    :param datavecs: list of lists - all the distributions to plot - lists inside of different length and
                        they contain the values of degrees for each node
    :param labels: list of str
    :param xlabel: str
    :param ylabel: str
    :param c: list of values for the colorbar
    :param max_x: max value among all datavecs

    :return: fig
    """
    n = len(datavecs)
    markers = (['_', 'v', '^', 'o', '+', 'x', 'd'] * 7)[:n]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.plasma(get_normalized_values(c))

    x_values = range(1, max_x + 1)
    avg_px = np.array([0] * max_x)

    for datavec, label, marker, color in zip(datavecs, labels, markers, colors):
        counter = [0] * max_x
        for value in datavec:
            counter[value - 1] += 1
        px = np.array([v / len(datavec) for v in counter])

        ax.loglog(x_values, px, marker=marker, label=label, color=color, linestyle=' ')

        avg_px = avg_px + px

    y = [v / n for v in avg_px[1:] if v > 0]
    x = range(2, len(y) + 2)

    logx = np.log(x)
    logy = np.log(y)
    [m, q] = np.polyfit(logx, logy, 1)
    print("m: %.4f, q: %.4f" % (m, q))

    if max_x == 28:
        max_x = 15
    y_fit = np.exp(m * logx[:max_x] + q)
    ax.plot(x[:max_x], y_fit, linestyle=':')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.5, .102), loc='lower left', ncol=5, mode="expand", borderaxespad=0.1)

    m = mpl.cm.ScalarMappable(cmap=mpl.cm.plasma)
    m.set_array(c)
    fig.colorbar(m)

    return fig


def plot_two_columns_dataframe(df, col_x, col_y1, col_y2):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    x, y1, y2 = df[col_x], df[col_y1], df[col_y2]

    df.plot(x=col_x, y=col_y1, ax=ax, kind='scatter')
    df.plot(x=col_x, y=col_y2, ax=ax2, color='r', kind='scatter', marker='+')

    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=70)

    return fig


''' DISTANCE ANALYSIS '''


def bfs(graph, vertex):
    queue = deque([vertex])
    level = {vertex: 0}
    parent = {vertex: None}

    while queue:
        v = queue.popleft()
        # for each neighbor of the pulled node
        for n in graph[v]:
            # if it wasn't visited yet
            if n not in level:
                # add it to the queue
                queue.append(n)
                # add its level
                level[n] = level[v] + 1
                # and its parent
                parent[n] = v
    return level, parent


def bfs_with_distance(graph, vertex, coords):
    queue = deque([vertex])  # create a queue
    level = {vertex: 0}      # create a dictionary for all the levels of the nodes
    parent = {vertex: None}  # create a dictionary to record parent of each node visited
    # create dict to record distances in km
    distances_bfs = {vertex: 0}
    distances_eu = {vertex: 0}
    # bfs_dist, eu_dist = [], []

    while queue:
        v = queue.popleft()
        # for each neighbor of the pulled node
        for n in graph[v]:
            # if it wasn't visited yet
            if n not in level:
                # add it to the queue
                queue.append(n)
                # add its level
                level[n] = level[v] + 1
                # and its parent
                parent[n] = v
                # compute distance from current node to vertex (core_node)
                distances_bfs[n] = distances_bfs[v] + geodesic(coords[v], coords[n]).km
                # compute euclidean distance from starting node to current one
                distances_eu[n] = geodesic(coords[vertex], coords[n]).km

    return level, parent, distances_bfs, distances_eu


def add_weights_to_network(net):

    w_dict = {}
    coords = nx.get_node_attributes(net, 'coords')
    for e in net.edges():
        w = geodesic(coords[e[0]], coords[e[1]]).km
        w_dict[e] = w

    nx.set_edge_attributes(net, 'weight', w_dict)


def compute_shortest_paths(net, start):

    coords = nx.get_node_attributes(net, 'pos')
    distances_net = nx.single_source_dijkstra_path_length(G=net, source=start, weight='weight')
    paths = nx.single_source_dijkstra_path(G=net, source=start)

    real_distances = compute_distances(paths, coords[start], coords, distances_net)

    return real_distances


def compute_distances(paths, coords1, coords, distances_net=None):
    real_distances = {}
    for k, v in paths.items():  # k = node, v = list of nodes in the path
        d_net = 0
        if distances_net is None:
            # compute network distance
            for i in range(len(v)):
                if i > 0:
                    d_net += geodesic(coords[v[i-1]], coords[v[i]]).km
        else:
            d_net = distances_net[k]
        if d_net > 0:
            # compute ral distance from core node
            real_distances[k] = (d_net, geodesic(coords1, coords[k]).km)
    return real_distances


def compute_paths(parent, core_node):

    paths = {}
    for k, v in parent.items():
        if v is not None:
            paths[k] = [core_node]
            while v != core_node:
                paths[k].append(v)
                v = parent[v]  # update the node to look at
            paths[k].append(k)

    return paths


def compute_near_and_far_distances_dictionaries(nodes, bfs_list, eu_list):

    max_eu = max(eu_list)
    threshold_near = 1 / 4 * max_eu
    threshold_far = 1 / 2 * max_eu  # vedere 3/4

    close_distances, far_distances = {}, {}

    for node, bfs, eu in zip(nodes, bfs_list, eu_list):
        if bfs < threshold_near:
            close_distances[node] = bfs
        elif bfs > threshold_far:
            far_distances[node] = bfs

    return close_distances, far_distances


def get_distances(paths, coords1, coords):
    """
    given the paths collect distances of all 3 types and return them

    :param paths:
    :param coords1:
    :param coords:
    :return:
    """
    real_distances = {'bfs': [], 'sp': [], 'ed': []}
    for k, v in paths.items():  # k = node, v = list of nodes in the path
        d_net_bfs, d_net_sp = 0, 0

        # compute network distance
        # BFS
        for i in range(len(v)):
            if i > 0:
                d_net_bfs += geodesic(coords[v[i - 1]], coords[v[i]]).km
        # shortest path
        # d_net_sp = distances_net[k]
        if d_net_sp > 0 or d_net_bfs > 0:
            # add all 3 to dictionary
            real_distances['bfs'].append(d_net_bfs)
            # real_distances['sp'].append(d_net_sp)
            real_distances['ed'].append(geodesic(coords1, coords[k]).km)

    return real_distances


def get_all_distances(net, core_node):

    # bfs info
    level, parent = bfs(net, core_node)
    paths = compute_paths(parent, core_node)

    # sp info
    coords = nx.get_node_attributes(net, 'coords')
    # distances_net = nx.single_source_dijkstra_path_length(G=net, source=core_node, weight='weight')
    # paths = nx.single_source_dijkstra_path(G=net, source=core_node)

    real_distances = get_distances(paths, coords[core_node], coords)

    return real_distances['bfs'], real_distances['ed']


def plot_distances_for_single_city(first, second, colors, labels, bins, xlabel, ylabel):

    mu1 = st.mean(first)
    sd1 = st.stdev(first)

    mu2 = st.mean(second)
    sd2 = st.stdev(second)

    params = [mu1, sd1, mu2, sd2]

    fig = plt.figure()

    sns.set(style="white", palette="muted", color_codes=True)
    sns.distplot(first, color=colors[0], label=labels[0]+' \u03BC=%.2f, \u03C3=%.2f' % (mu1, sd1), bins=bins[0])
    sns.distplot(second, color=colors[1], label=labels[1]+' \u03BC=%.2f, \u03C3=%.2f' % (mu2, sd2), bins=bins[1])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.tight_layout()

    return fig, params


def plot_distances_fractions(fractions):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.set(style="white", palette="muted", color_codes=True)
    for k, v in fractions.items():
        sns.distplot(v, hist=False, kde=True, label=k)

    ax.set_xlabel('Fraction')
    ax.set_ylabel('Density')
    ax.grid()
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=3, mode="expand", borderaxespad=0.1)

    return fig


def get_peripheral_nodes(net, coords, distances_eu, json_file=None):

    if json_file is not None:
        return load_json(json_file)

    peripheral_dict = {node: True for node in distances_eu.keys()}

    for edge in net.edges():
        if edge[0] in peripheral_dict and edge[1] in peripheral_dict:
            # a, b = int(edge[0]), int(edge[1])
            if distances_eu[edge[0]] > distances_eu[edge[1]]:
                peripheral_dict[edge[1]] = False
            if distances_eu[edge[0]] < distances_eu[edge[1]]:
                peripheral_dict[edge[0]] = False

    peripheral_list = [node for node, flag in peripheral_dict.items() if flag is True]

    print('Reached nodes = %d, first peripherals = %d' % (len(peripheral_dict), len(peripheral_list)))
    max_eu_distance = max(distances_eu.values())
    d = max_eu_distance/10
    for v in peripheral_list:
        for w in peripheral_list:
            if v != w:
                if geodesic(coords[v], coords[w]).km < d:
                    # compare the distances of the two nodes from the center
                    if distances_eu[v] < distances_eu[w]:
                        peripheral_dict[v] = False

    return peripheral_dict


''' FREQUENCY ANALYSIS '''


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls2rgb(h, l, s)
    return rgb2hex(int(r * 255), int(g * 255), int(b * 255))


def darken_color(r, g, b, factor=0.1):
    return adjust_color_lightness(r, g, b, 1 - factor)


def plot_bar_frequencies(d, color, title):

    colors = [color]*len(d)
    d_morning = dict(list(d.items())[:12])
    d_afternoon = dict(list(d.items())[12:])
    m_morning = int(max(d_morning, key=d_morning.get))
    m_afternoon = int(max(d_afternoon, key=d_afternoon.get))

    # darken bars for morning peak hour and afternoon one
    h = mc.to_hex(color)
    r, g, b = hex_to_rgb(h)  # hex to rgb format
    darker = darken_color(r, g, b, 0.5)

    colors[m_morning] = darker
    colors[m_afternoon] = darker

    fig, ax = plt.subplots()
    ax.bar(*zip(*d.items()), color=colors, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Hours')
    ax.set_ylabel('Number of vehicles')
    fig.tight_layout()

    return fig


def plot_distribution_frequency(f_list, color):

    mu1 = st.mean(f_list)
    sd1 = st.stdev(f_list)

    fig = plt.figure()

    sns.set_style("white")
    # kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
    # sns.set(style="white", palette="muted", color_codes=True)
    sns.distplot(f_list, color=color, label="\u03BC=%.2f, \u03C3=%.2f" % (mu1, sd1), bins=len(f_list))

    plt.xlabel('Number of vehicles per hour')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')

    return fig


def plot_bars_mu_st(labels, mus, sts, ylabel, title=None, type=3, feature=None, feature_label=None, color=None):

    types_and_colors = get_types_of_transport_and_colors()
    dict_number_types = get_dict_number_type_of_transport()

    type_of_transport = dict_number_types[type]
    if color is None:
        color = types_and_colors[type]
    h = mc.to_hex(color)
    r, g, b = hex_to_rgb(h)  # hex to rgb format
    darker = darken_color(r, g, b, 0.5)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, mus, width, label='\u03BC', color=color, alpha=0.5, edgecolor='k')
    ax.bar(x + width / 2, sts, width, label='\u03C3', color=darker, alpha=0.5, edgecolor='k')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    if title is None:
        ax.set_title(type_of_transport + ' transport network', color=color)
    else:
        ax.set_title(title, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70)

    if feature is not None:
        ax1 = ax.twinx()
        x = np.arange(len(labels))
        y = feature
        ax1.plot(x, y, color='k', label=feature_label, marker='o')
        if feature_label == 'Population':
            y1_labels = ['{}'.format(int(i)) + 'M' for i in ax1.get_yticks() / 10**6]
            ax1.set_yticklabels(y1_labels)
            ax1.set_ylabel('Millions of inhabitants')
        elif feature_label == 'Area':
            y1_labels = ['{}'.format(round(i, 1)) + 'K' for i in ax1.get_yticks() / 10 ** 3]
            ax1.set_yticklabels(y1_labels)
            ax1.set_ylabel('Thousands of km')
        ax1.legend(loc=2)

    ax.legend(loc=9)
    fig.tight_layout()

    return fig


def plot_network_with_nodes_colormap(net, measure_name, max, title):
    """
    Plot city network with nodes colored depending on centrality measure (NOTE: not all nodes!)

    :param net: networkx obj
    :param measure_name: str - type of centrality measure
    :param max: max value
    :param title: str - title of the figure

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


def set_net_attributes_and_plot(net, city, attr_name, attr_dict, type, dir_plots, type_of_transport):

    # add attributes to nodes in the network
    nx.set_node_attributes(net, attr_name, attr_dict[type])
    max_peak = max(attr_dict[type].values())

    # plot network with colors based on the value of the attribute
    title = city + ' '+attr_name+' ' + type_of_transport + ' transport network'
    fig = plot_network_with_nodes_colormap(net, attr_name, max_peak, title)
    if fig is not None:
        # fig.show()
        fig_name = dir_plots + type_of_transport + '_'+attr_name+'.png'
        fig.savefig(fig_name, bbox_inches='tight')


def get_mean_and_peak_hour(frequency_dict, route_type):

    peak_hour = max(frequency_dict[route_type], key=frequency_dict[route_type].get)

    # save info about peak hour of the city and the mean value
    mean = st.mean(frequency_dict[route_type].values())
    values = frequency_dict[route_type].values()
    closest_value = min(values, key=lambda list_value: abs(list_value - mean))
    # closest_key = int([k for k, v in frequency_dict[i].items() if v == closest_value])
    closest_key = str(list(frequency_dict[route_type].keys())
                      [list(frequency_dict[route_type].values()).index(closest_value)])

    return peak_hour, closest_key
