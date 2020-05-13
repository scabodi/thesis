from gtfspy import gtfs, networks, mapviz
import matplotlib.pyplot as plt


def load_gtfs(city):
    """
    Load gtfs object from sqlite file

    :param city: str - name of the city
    :return: g: gtfs object of the specified city
    """
    imported_database_path = "data/"+city+"/week.sqlite"

    g = gtfs.GTFS(imported_database_path)

    return g


def load_coords_stops(g):
    """
    Given a gtfs obj, load coordinates of every stop

    :param g: gtfs object
    :return: coords: dict - keys = stop_I, values = tuples of coordinates
    """
    stops = g.stops()
    coords = {}
    for stop in stops.itertuples():
        coords[stop.stop_I] = (stop.lat, stop.lon)

    return coords


def plot_gtfs(g, city):
    """
    Plot gtfs object over a map

    :param g: gtfs object
    :param city: str - name of the city to plot
    """

    ax = mapviz.plot_route_network_from_gtfs(g, scalebar=True)
    mapviz.plot_all_stops(g, ax)

    plt_name = './data/' + city + '/plot_net.png'
    plt.savefig(plt_name)
    plt.close()


def create_network(g):
    """
    Create network from gtfs obj
    
    :param g: gtfs obj
    :return: net: networkx object
    """

    net = networks.combined_stop_to_stop_transit_network(gtfs=g)

    return net
