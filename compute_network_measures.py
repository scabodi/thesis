import network_functions as nf

if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types = nf.get_types_of_transport_and_colors()
    centrality_dict, measures_dict, degrees = {}, {}, {}
    additional_measures = {'<r>': [], '<l>': [], '<knn>': []}

    tol = 10 ** -1

    degree_betweenness = {city: {} for city in cities}

    prefix_json = 'results/all/json/'
    dump = True  # Put to True in case you want to save again the results into json format files

    ''' Computations over each city '''
    for city in cities:

        print('Processing ' + city + ' ...')

        ''' CREATE NETWORK - undirected and unweighted and add it to the proper dictionary '''
        net = nf.create_network(city, types=types)

        # ''' plot the network without difference between types of transport '''
        # nf.plot_network(city, net)
        #
        # ''' Compute basic measures for specific city '''
        # measures_dict[city] = {}
        # nf.compute_measures(net, measures_dict[city])
        # # print(measures_dict[city])
        #
        # ''' Compute degree distribution for each city '''
        # # degree -- list of nodes degree
        # degrees[city] = [v for k, v in net.degree().items()]
        #
        # ''' Compute additional measures '''
        # r = nf.get_assortativity(net)
        # additional_measures['<r>'].append(r)
        #
        # s = nf.get_avg_shortest_path_length(net)
        # additional_measures['<l>'].append(s)
        #
        # knn = nf.get_avg_degree_connectivity(net)
        # additional_measures['<knn>'].append(knn)
        #
        # ''' Compute centrality measures '''
        # [degree, betweenness, closeness, eigenvector] = nf.get_centrality_measures(net, tol)
        # centrality_dict[city] = {
        #     'degree': list(degree),
        #     'betweenness': list(betweenness),
        #     'closeness': list(closeness),
        #     'eigenvector': list(eigenvector),
        # }
        #
        # file_name = './results/'+city+'/centrality_measures.json'
        # nf.dump_json(file_name, centrality_dict[city])

        nf.set_degree_betweenness_dict_for_city(net, degree_betweenness, city)
        print(degree_betweenness[city])

    if dump:
        # nf.dump_json(prefix_json+'degrees.json', degrees)
        # nf.dump_json(prefix_json+'centrality_measures.json', centrality_dict)
        # nf.dump_json(prefix_json+'additional_measures.json', additional_measures)
        nf.dump_json(prefix_json+'degree_betweenness.json', degree_betweenness)
