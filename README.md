# Thesis

## Introduction
Public Transport Network analysis of 27 cities around the world.

This project has been done at Aalto University (Espoo, Finland), under the supervision of professor Jari Saramaki.

The work is developed in three main phases:
1. Network analysis and characterization
    * Basic measures: nodes, edges, diameter, density, average clustering coefficient
    * Additional measures: assortativity, average path length, average degree, degree distribution
    * Centrality measures: betweenness, closeness, degree and eigenvector 
2. Distance analysis 
    * PTN efficiency evaluated by shortest path length vs. Euclidean distance
3. Frequency analysis
    * Frequency distribution of vehicles in a typical day
    
## Folders

### Data 
The *data* folder includes the data files necessary to perform the analysis. All the data gathered for the PTNs analysis is available at http://transportnetworks.cs.aalto.fi/.
Each city has its own subfolder containing most of the available files (not all here on GitHub due to their dimention).

### Results

The *results* folder contains all the results obtained by the analysis of the dataset. Each city has its own folder and an additional folder, called *all*,
containing some overall statistics. 

For each city there are several subfolders:
* **centrality_measures**: contains the information about the 4 centrality measures in different formats (json, distributions, scatter plots and network representations)
* **connected_components/plots**: gives a visual representation of each connected component in the current city network
* **distance_analysis**: offers both json and visual representations of the results obtained for the distance analysis
* **frequency_analysis**: offers both json and visual representations of the results obtained for the frequency analysis, this time divided by type of transport

### Venv

The *venv* folder contains the files related to the Python virtual environment and all Python packages necessary to run the project.

## Files

All the Python files present in the project have self-explaining names that describe at high level their purpose. 
