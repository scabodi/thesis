import network_functions as nf
import os.path as path

if __name__ == '__main__':

    cities = nf.get_list_cities_names()
    types = nf.get_types_of_transport_and_colors()
    type_of_transport = nf.get_types_of_transport_names()

    for city in cities:
        for type in type_of_transport:
            edges_file = 'data/'+city+'/network_'+type+'.csv'
            if path.exists(edges_file):
                # create the network
                net = nf.create_network(city, types, edges_file)
                # make here computations
