import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
from collections import Counter
from grakel import Graph, WeisfeilerLehman, GraphKernel, VertexHistogram
from grakel.graph_kernels import WeisfeilerLehmanOptimalAssignment
from sklearn import svm, model_selection
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import lime.lime_tabular
from sklearn.metrics import accuracy_score
import seaborn as sns
from re import *
from itertools import combinations



def show_dictionary_images_and_networkx(filename):

    # Load the JSON file
    with open(filename, "r") as file:
        data = json.load(file)

    # For each entry in the dictionary
    for name, info in data.items():
        # Convert the NetworkX JSON object back into a NetworkX graph
        networkx_obj = nx.readwrite.json_graph.node_link_graph(info["networkx_obj"])
        image_filename = info["image"]

        # Draw the NetworkX graph
        plt.subplot(1, 2, 1)
        pos = nx.spring_layout(networkx_obj)
        nx.draw(networkx_obj, pos, with_labels=True, node_color='lightblue', 
                node_size=100, font_size=8, font_weight='bold', edge_color='black', 
                width=0.5, arrowsize=10)

        # Load and display the image
        plt.subplot(1, 2, 2)
        extensions = ['', '.png', '.jpg']
        for ext in extensions:
            try:
                img = mpimg.imread("Feynman_images/" + image_filename + ext)
                break
            except FileNotFoundError:
                continue

        plt.imshow(img)
        plt.axis('off')

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.2)

        # Show the combined figure
        plt.show()


def wl_relabel(G):
    # Initialize with node degrees
    new_labels = dict(nx.degree(G))

    # Prepare data structure for unique labels
    label_lookup = dict()
    label_counter = Counter(new_labels.values())
    new_labels_values = sorted(label_counter.keys())
    for label in new_labels_values:
        label_lookup[label] = str(label)
        if label_counter[label] > 1:
            label_lookup[label] += 'a'

    # Assign initial unique labels
    for node in G.nodes:
        new_labels[node] = label_lookup[new_labels[node]]

    # Prepare data structure for relabeling
    labels = {}

    # Weisfeiler-Lehman relabeling procedure
    for it in range(len(G.nodes)):
        for node in G.nodes:
            neighbors = G.neighbors(node)
            labels_sorted = sorted([new_labels[neigh] for neigh in neighbors])
            labels[node] = new_labels[node] + ''.join(labels_sorted)

        label_counter = Counter(labels.values())
        new_labels_values = sorted(label_counter.keys())
        label_lookup = dict()

        for label in new_labels_values:
            if label not in label_lookup:
                if label_counter[label] > 1:
                    label_lookup[label] = str(it + 1) + 'a'
                else:
                    label_lookup[label] = str(it + 1)

        for node in G.nodes:
            new_labels[node] = label_lookup[labels[node]]

    return new_labels

def networkx_obj_to_grakel(networkx_obj):
    # Construct adjacency dictionary for GraKeL
    adjacency_dict = {node['id']: [] for node in networkx_obj['nodes']}
    for link in networkx_obj['links']:
        adjacency_dict[link['source']].append(link['target'])
        if not networkx_obj['directed']:
            adjacency_dict[link['target']].append(link['source'])

    # Construct node label dictionary for GraKeL, splitting id on underscore
    labels_dict = {node['id']: node['id'].split('_')[0] for node in networkx_obj['nodes']}
    # Modify labels_dict such that 'top' becomes 't'
    for node_id, label in labels_dict.items():
        if label == 'top':
            labels_dict[node_id] = 't'
    # Now, create the GraKeL Graph
    grakel_graph = Graph(adjacency_dict, node_labels=labels_dict)
    return grakel_graph


def print_graph_info(grakel_graph):
    node_labels = grakel_graph.get_labels(purpose="any", label_type="vertex")
    edges = grakel_graph.get_edges(purpose="dictionary")

    print("Node labels: ", node_labels)
    print("Number of nodes: ", len(node_labels))
    
    num_edges = len(edges)

    print("Number of edges: ", int(num_edges/2))
    print(edges)
    print(node_labels)

# Load the JSON file
def convert_dict_to_grakel_graphs(data_dict):
    with open(data_dict, 'r') as f:
        data_dict = json.load(f)
    grakel_graphs = []
    for _, val in data_dict.items():
        # Convert the NetworkX object to GraKeL Graph
        converted_dict = networkx_obj_to_grakel(val['networkx_obj'])
        
        # Append the GraKeL Graph to the list
        grakel_graphs.append(converted_dict)
        
    return grakel_graphs

def compute_wl_kernel(grakel_graphs, h=3):
    # Initialize the WeisfeilerLehmanOptimalAssignment kernel
    gk = WeisfeilerLehmanOptimalAssignment(n_iter = h, normalize=True)

    # Fit and transform the kernel on the input graph list
    K = gk.fit_transform(grakel_graphs)

    return K

def parse_input(grakel_graphs, h):
    # Initialize the WeisfeilerLehmanOptimalAssignment kernel
    gk = WeisfeilerLehmanOptimalAssignment(n_iter = h, normalize=False)

    # Fit and transform the kernel on the input graph list
    gk.fit_transform(grakel_graphs)
    features = gk.parse_input(grakel_graphs)
    return features

# def parse_input_vh(grakel_graphs):
#     gk = VertexHistogram()
#     gk.fit_transform(grakel_graphs)
    
#     return 

def representation(grakel_graphs, h):
    # Initialize the WeisfeilerLehmanOptimalAssignment kernel
    gk = WeisfeilerLehmanOptimalAssignment(n_iter = h, normalize=True)

    # Fit and transform the kernel on the input graph list
    gk.fit_transform(grakel_graphs)
    return gk._inv_labels

def representation_vh(grakel_graphs):
    gk = VertexHistogram()
    gk.fit_transform(grakel_graphs)
    return gk._inv_labels

def get_classes_from_json(json_file):
    with open(json_file, 'r') as f:
        data_dict = json.load(f)

    classes = []
    for key, val in data_dict.items():
        classes.append(val['class_name'])
        
    return classes

def feynman_SVM(grakel_graphs, class_list, test_size=0.3, cv=5):
    # Initialize a Weisfeiler-Lehman subtree kernel
    # Calculate the kernel matrix.
    K = compute_wl_kernel(grakel_graphs)

    # Create numpy array from the classes list
    classes = np.array(class_list)
    # Train a SVM using a One-vs-One strategy
    clf = OneVsOneClassifier(svm.SVC(kernel='precomputed'))
    clf.fit(K, classes)

    # Cross-validation
    scores = model_selection.cross_val_score(clf, K, classes, cv=cv)
    print("Cross-validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    return clf


def random_forest_classifier(grakel_graphs, class_list):
    # Initialize a Weisfeiler-Lehman subtree kernel
    # Calculate the kernel matrix.
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    # Create numpy array from the classes list
    classes = np.array(class_list)

    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(my_list, classes)

    # Evaluate the model
    scores = model_selection.cross_val_score(rf, my_list, classes, cv=5)
    print("Cross-validation accuracy random-forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def lime_explainer(grakel_graphs, class_list, index):
    feature_names_list = ["strong", "weak", "electromagnetic"]
    # Create numpy array from the classes list
    classes = np.array(class_list)
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(my_list, classes)

    explainer = lime.lime_tabular.LimeTabularExplainer(my_list, class_names = feature_names_list, discretize_continuous=True, random_state=42)
    # An instance to justify the model's selection
    i = index
    instance = my_list[i]

    # Generate an explanation
    exp = explainer.explain_instance(instance, rf.predict_proba, num_features=20, top_labels=3)

    # Visualize the explanation for each class
    for i in range(3):
        print('Explanation for class %s:' % i)
        exp.show_in_notebook(show_table=True, labels=[i])
    # Get the explanations as a list
    # Define the classes
    classes_list = [0, 1, 2]  # Update this to match your actual classes
    feature_weights_output = []
    # For each class, get and print the explanations
    for class_id in classes_list:
        explanation_list = exp.as_list(label=class_id)
        for feature_weight in explanation_list:
            feature_weights_output.append([feature_weight])
    return feature_weights_output

def lime_explainer_vh(grakel_graphs, class_list, index):
    feature_names_list = ["strong", "weak", "electromagnetic"]
    # Create numpy array from the classes list
    classes = np.array(class_list)
    my_list = parse_input_vh(grakel_graphs, 1)
    my_list = np.array(my_list)
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(my_list, classes)

    explainer = lime.lime_tabular.LimeTabularExplainer(my_list, class_names = feature_names_list, discretize_continuous=True, random_state=42)
    # An instance to justify the model's selection
    i = index
    instance = my_list[i]

    # Generate an explanation
    exp = explainer.explain_instance(instance, rf.predict_proba, num_features=20, top_labels=3)

    # Visualize the explanation for each class
    for i in range(3):
        print('Explanation for class %s:' % i)
        exp.show_in_notebook(show_table=True, labels=[i])
    # Get the explanations as a list
    # Define the classes
    classes_list = [0, 1, 2]  # Update this to match your actual classes
    feature_weights_output = []
    # For each class, get and print the explanations
    for class_id in classes_list:
        explanation_list = exp.as_list(label=class_id)
        for feature_weight in explanation_list:
            feature_weights_output.append([feature_weight])
    return feature_weights_output

def plot_rf_error(grakel_graphs, class_list):
    # Initialize a Weisfeiler-Lehman subtree kernel
    # Calculate the kernel matrix.
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(my_list, classes)

    # Cross-validation of the model
    cv_scores = model_selection.cross_val_score(rf, my_list, classes, cv=6)

    # Calculate the error rates
    errors = [1 - cv for cv in cv_scores]

    # Plot the error rates
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(errors)), errors, marker='o')
    plt.title('Error Rate for each CV fold')
    plt.xlabel('CV fold')
    plt.ylabel('Error rate')
    plt.show()


def plot_confusion_matrix(grakel_graphs, class_list):
    # Initialize a Weisfeiler-Lehman subtree kernel
    # Calculate the kernel matrix.
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(my_list, classes, test_size=0.4, random_state=42)

    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def grakel_to_networkx(G):
    g_nx = nx.Graph()

    for u, v, data in G.edges(data=True):
        g_nx.add_edge(u, v, **data)

    for n, data in G.nodes(data=True):
        g_nx.nodes[n].update(data)

    return g_nx

def find_substructure(graph, labels):
    subgraph_nodes = []
    
    for node in graph.nodes(data=True):
        if node[1]['label'] in labels:
            subgraph_nodes.append(node[0])
            labels.remove(node[1]['label'])

    # Find the edges that are part of the substructure
    subgraph_edges = [edge for edge in graph.edges() if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes]
    
    return subgraph_nodes, subgraph_edges

def visualize_substructure(graph, subgraph_nodes, subgraph_edges):
    pos = nx.spring_layout(graph)  # position layout
    plt.figure(figsize=(8, 6))

    # draw all nodes and edges in the graph in light color
    nx.draw_networkx(graph, pos, node_color='blue', edge_color='black')
    
    # draw the nodes and edges of the subgraph in dark color
    nx.draw_networkx_nodes(graph, pos, nodelist=subgraph_nodes, node_color='red')
    nx.draw_networkx_edges(graph, pos, edgelist=subgraph_edges, edge_color='red')
    nx.draw_networkx_labels(graph, pos)
    plt.show()

def find_substructure(graph, labels):
    # Define substructure finding for small number of labels
    def modified_method(labels):
        subgraph_nodes = set()
        subgraph_edges = set()

        # Check all edges in the graph
        for u, v, data in graph.edges(data=True):
            edge_labels = [graph.nodes[u]['label'], graph.nodes[v]['label']]
            if all(any(label.strip("_") in edge_label for edge_label in edge_labels) for label in labels):
                subgraph_nodes.update([u, v])
                subgraph_edges.add((u, v))
        return subgraph_nodes, subgraph_edges

    # If number of labels is small, use modified method
    if len(labels) <= 3:
        return modified_method(labels)

    # If number of labels is greater than 3, use exhaustive search
    else:
        priority_labels = ['Z', 'W', 'g', 'gamma']

        # Sort labels according to the priority list
        labels.sort(key=lambda x: priority_labels.index(x) if x in priority_labels else len(priority_labels))

        all_nodes = [node for node in graph.nodes(data=True)]
        for nodes_comb in combinations(all_nodes, 4):
            subgraph_nodes = [node[0] for node in nodes_comb]
            subgraph = graph.subgraph(subgraph_nodes)
            node_labels = [data['label'] for _, data in subgraph.nodes(data=True)]

            if set(labels) == set(node_labels) and nx.is_connected(subgraph):
                subgraph_edges = [edge for edge in graph.edges() if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes]
                return subgraph_nodes, subgraph_edges

        # If no suitable subgraph found
        return set(), set()






def visualize_substructure(graph, subgraph_nodes, subgraph_edges):
    pos = nx.spring_layout(graph)  # position layout
    plt.figure(figsize=(8, 6))

    # draw all nodes and edges in the graph in light color
    nx.draw_networkx(graph, pos, node_color='blue', edge_color='black')
    
    # draw the nodes and edges of the subgraph in dark color
    nx.draw_networkx_nodes(graph, pos, nodelist=subgraph_nodes, node_color='red')
    nx.draw_networkx_edges(graph, pos, edgelist=subgraph_edges, edge_color='red')
    nx.draw_networkx_labels(graph, pos)
    plt.show()

def grakel_to_networkx(grakel_graph):
    # Initialize a new NetworkX graph
    networkx_graph = nx.Graph()

    # Get the nodes and edges from the GraKeL graph
    node_labels = grakel_graph.get_labels(purpose='dictionary', label_type='vertex')

    # Add the nodes and their labels to the NetworkX graph
    for node, label in node_labels.items():
        networkx_graph.add_node(node, label=label)
        print(label)

    # Add the edges to the NetworkX graph
    for ((node1, node2), weight) in grakel_graph.get_edges(purpose="dictionary"):
        networkx_graph.add_edge(node1, node2)

    return networkx_graph

def get_substructure_labels(feature_vector):
    feature_map = representation(graphs, 1)[1]
    print(feature_map)
    node_mappings = representation(graphs, 1)[0]
    print((node_mappings))
    inverse_node_mappings = {v: k for k, v in node_mappings.items()}
    # Get indices of non-zero entries in the feature vector, increment each index by 22
    positions = [index+22 for index, value in enumerate(feature_vector) if value > 0]
    
    # Translate positions to the corresponding keys in the feature map
    keys = [k for k, v in feature_map.items() if v in positions]
    
    # Extract the node labels from the keys and translate them back to the original labels
    substructure_labels = []
    for key in keys:
        # extract all numbers from the key and convert them to integers
        node_indices = [int(s) for s in findall(r'\b\d+\b', key)]
        labels = [inverse_node_mappings[i] for i in node_indices]
        substructure_labels.append(labels)

    return substructure_labels



json_file = 'dictionary.json'
classes = get_classes_from_json(json_file)
graphs = convert_dict_to_grakel_graphs("dictionary.json")


