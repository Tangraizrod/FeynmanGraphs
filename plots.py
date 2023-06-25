from functions import *


plot_hyperparameter_tuning(graphs, classes, depth_range, trees_range)


show_dictionary_images_and_networkx("dictionary.json", -1) #last graph added

process_graph(-1, "dictionary.json") #graphs labels plot

lime_explainer(graphs, classes, 4) #lime explainer