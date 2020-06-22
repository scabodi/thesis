import network_functions as nf
import matplotlib.pyplot as plt
import generic_plot_functions as pf

if __name__ == '__main__':

    cities = nf.get_list_cities_names()

    area_population_file = 'results/all/json/area_population.json'

    ''' Load info about areas and populations for each city and plot them '''
    area_population_dict = nf.load_json(area_population_file)
    # print(area_population_dict)
    # print(sorted(area_population_dict.items()))

    areas = nf.get_list_sorted_values('area', area_population_dict)
    populations = nf.get_list_sorted_values('population', area_population_dict)

    y_values = [areas, populations]
    colors = ['r', 'b']
    labels = ['areas', 'population']

    fig1 = pf.plot_bars_with_subplots(2, 1, cities, y_values, colors, labels)
    fig_name = './results/all/plots/basic_measures/area_population_plot.png'
    plt.savefig(fig_name)
    plt.close()


