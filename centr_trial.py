import network_functions as nf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


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

    styles = (['-', '--', '-.', ':']*7)[:len(datavecs)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_xlim(left=0)
    colors = plt.cm.plasma(nf.get_normalized_values(c))

    for datavec, label, style, color in zip(datavecs, labels, styles, colors):
        sorted_datavec = [x for x in sorted(datavec) if x > 10**-4]
        cdf = np.zeros(len(sorted_datavec))

        for i, val in enumerate(sorted_datavec, 0):
            cdf[i] = val
            if i > 0 and val > 0:
                cdf[i] += cdf[i-1]
        sum_tot = cdf[len(cdf)-1]
        ccdf = [1-(x/sum_tot) for x in cdf]
        ax.loglog(sorted_datavec, ccdf, linestyle=style, label=label, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=3, mode="expand", borderaxespad=0.1)
    # plt.yscale('log')

    m = mpl.cm.ScalarMappable(cmap=mpl.cm.plasma)
    m.set_array(c)
    fig.colorbar(m)

    return fig


if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types = nf.get_types_of_transport_and_colors()

    centrality_dict = {}

    area_population_file = 'results/all/json/area_population.json'
    area_population_dict = nf.load_json(area_population_file)
    areas = nf.get_list_sorted_values('area', area_population_dict)
    populations = nf.get_list_sorted_values('population', area_population_dict)

    ''' Computations over each city '''
    for city in cities:

        print('Processing ' + city + ' ...')

        # net = nf.create_network(city, types=types)
        # nf.plot_network(city, net)
        #
        # ''' Load centrality measures for specific city '''
        json_path = 'results/'+city+'/centrality_measures.json'
        centrality_measures = nf.load_json(json_path)
        # values = [np.array(v) for k, v in centrality_measures.items()]

        # Save in the dict for each type of centrality a list of lists of values to plot later
        for k, v in centrality_measures.items():
            if k not in centrality_dict:
                centrality_dict[k] = []
            centrality_dict[k].append(list(v))

        # ''' Plot ccdf of all measures of centrality considered for current city '''
        # datavecs = list(centrality_measures.values())
        # labels = list(centrality_measures.keys())
        # xlabel = 'measure'
        # ylabel = 'P(measure)'
        # fig = nf.plot_ccdf(datavecs, labels, xlabel, ylabel)
        # fig_name = './results/'+city+'/ccdf_centrality_measures.png'
        # fig.savefig(fig_name)
        #
        # ''' Plot all distributions together '''
        # x_values = net.nodes()
        # y_values = list(centrality_measures.values())
        # fig = nf.plot_multiple_lines(x_values, y_values, labels, xlabel, ylabel)
        # fig_name = './results/' + city + '/lin_log_distributions.png'
        # fig.savefig(fig_name)
        #
        # for k, v in centrality_measures.items():
        #     sorted_values = v.copy()
        #     sorted_values.sort()
        #
        #     ''' Plot distribution of current centrality measure'''
        #     fig1 = nf.plot_distribution(list(net.nodes()), list(sorted_values), "nodes", k)
        #     fig_name = './results/'+city+'/distr_plot_'+k+'_centrality.png'
        #     fig1.savefig(fig_name)
        #
        #     ''' Plot network with 20% of significant nodes with color depending on value of centrality '''
        #     fig2 = nf.plot_network_with_centrality_measures(net, v, k)
        #     fig_name = './results/'+city+'/network_'+k+'_centrality.png'
        #     fig2.savefig(fig_name)
        #     plt.close('all')

    ''' Plot all distributions of the same centrality type for all the cities coloring with colormap the 
    line depending on the area of the city and the population of it  '''
    for measure, values in centrality_dict.items():
        xlabel = measure
        ylabel = 'P('+measure+')'
        labels = cities
        fig = plot_multiple_ccdf_with_colorbar(values, labels, xlabel, ylabel, c=areas)
        fig_name = './results/all/plots/centrality_measures/ccdf_'+measure+'_centrality_area.png'
        fig.show()
        # fig.savefig(fig_name, bbox_inches='tight')

        fig = plot_multiple_ccdf_with_colorbar(values, labels, xlabel, ylabel, c=populations)
        fig_name = './results/all/plots/centrality_measures/ccdf_' + measure + '_centrality_population.png'
        fig.show()
        # fig.savefig(fig_name, bbox_inches='tight')
