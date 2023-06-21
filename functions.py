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

def representation(grakel_graphs, h):
    # Initialize the WeisfeilerLehmanOptimalAssignment kernel
    gk = WeisfeilerLehmanOptimalAssignment(n_iter = h, normalize=True)

    # Fit and transform the kernel on the input graph list
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

json_file = 'dictionary.json'
classes = get_classes_from_json(json_file)
graphs = convert_dict_to_grakel_graphs("dictionary.json")


