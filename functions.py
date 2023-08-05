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
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lime.lime_tabular
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut

def show_dictionary_images_and_networkx(filename, index):
    """
    Loads data from a JSON file, then visualizes both a network graph and an image corresponding to a specific 
    entry determined by the provided index.

    The JSON file should contain a dictionary where keys are names of entries and values are 
    dictionaries with two keys:
        - 'networkx_obj': serialized NetworkX graph in node-link data format.
        - 'image': the filename of an image (without extension) located in the 'Feynman_images' directory.

    The function attempts to load the image with several extensions (.png, .jpg, .jpeg), displaying the first 
    one it finds.

    Parameters
    ----------
    filename : str
        The path to the JSON file.
    index : int
        The index of the dictionary entry to be visualized.

    Raises
    ------
    FileNotFoundError
        If the JSON file or image file specified by 'image' cannot be found.

    Notes
    -----
    The function requires matplotlib.pyplot (as plt), networkx (as nx), matplotlib.image (as mpimg), 
    and json libraries to be imported.

    The function uses subplots to display both the graph and the image side by side.

    The NetworkX graph is drawn using the spring layout.

    Examples
    --------
    >>> show_dictionary_images_and_networkx('data.json', 2)
    """

    # Load the JSON file
    with open(filename, "r") as file:
        data = json.load(file)

    # Get the specific entry based on the index
    name = list(data.keys())[index]
    info = data[name]

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
    extensions = ['', '.png', '.jpg', '.jpeg']
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
    """
    Converts a NetworkX graph object into a GraKeL graph.

    The function first constructs an adjacency dictionary, followed by a node label dictionary for GraKeL.
    The node labels are created by splitting the node id on underscore. The special case for 'top' label is 
    considered, which becomes 't' in the resulting graph.

    Parameters
    ----------
    networkx_obj : NetworkX graph
        The NetworkX graph to convert. The graph is expected to be in node-link data format.
        It should contain 'nodes' and 'links', where 'nodes' are a list of dictionaries with an 'id' field, 
        and 'links' are a list of dictionaries with 'source' and 'target' fields. The graph should also have a 
        'directed' field indicating if it's a directed graph.

    Returns
    -------
    grakel_graph : GraKeL graph
        The converted graph in GraKeL format.

    Raises
    ------
    KeyError
        If 'nodes', 'links', or 'directed' are not present in the NetworkX graph object.

    Examples
    --------
    >>> G = nx.erdos_renyi_graph(10, 0.5)
    >>> G = nx.readwrite.json_graph.node_link_data(G)
    >>> networkx_obj_to_grakel(G)

    Note
    ----
    GraKeL is a Python library for graph kernels, i.e., methods of comparing graphs in machine learning.
    """
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
    """
    Prints various information about a GraKeL graph.

    The function prints the node labels, total number of nodes, number of edges (undirected), 
    and adjacency list of the graph.

    Parameters
    ----------
    grakel_graph : GraKeL graph
        The input graph about which to print information.

    Examples
    --------
    >>> G is a networkx graph
    >>> grakel_graph = networkx_obj_to_grakel(G)
    >>> print_graph_info(grakel_graph)

    Notes
    -----
    The function assumes that the graph is undirected. If the graph is directed, the number of edges will be 
    overestimated.
    """
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
    """
    Reads a JSON file and converts all NetworkX graphs within it to GraKeL graphs.

    The input JSON file should contain a dictionary where each key-value pair corresponds to an entry. 
    Each value must be another dictionary that contains a 'networkx_obj' key, whose value is a NetworkX graph 
    in node-link data format.

    The function returns a list of GraKeL graphs.

    Parameters
    ----------
    data_dict : str
        Path to the JSON file containing the data dictionary.

    Returns
    -------
    grakel_graphs : list
        A list of converted GraKeL graphs.

    Raises
    ------
    FileNotFoundError
        If the JSON file specified by 'data_dict' cannot be found.
    KeyError
        If 'networkx_obj' is not found in the data dictionary.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs:

    {
        "graph1": {"networkx_obj": <node-link data representation of a graph>},
        "graph2": {"networkx_obj": <node-link data representation of another graph>}
        ...
    }

    You can convert all these graphs to GraKeL format with the following code:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    """
    with open(data_dict, 'r') as f:
        data_dict = json.load(f)
    grakel_graphs = []
    for _, val in data_dict.items():
        # Convert the NetworkX object to GraKeL Graph
        converted_dict = networkx_obj_to_grakel(val['networkx_obj'])
        
        # Append the GraKeL Graph to the list
        grakel_graphs.append(converted_dict)
        
    return grakel_graphs

def compute_wl_kernel(grakel_graphs, h=1):
    """
    Computes the Weisfeiler-Lehman Optimal Assignment (WL) kernel on a list of GraKeL graphs.

    The WL kernel is a graph kernel that computes the similarity between graphs based on their node labels.
    It involves an iterative procedure, controlled by the parameter 'h', that aggregates information 
    from a node's neighbors to update its label. The kernel is normalized by default.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects on which to compute the kernel.
    h : int, optional
        The number of iterations for the Weisfeiler-Lehman procedure, default is 1.

    Returns
    -------
    K : numpy.ndarray
        The computed Weisfeiler-Lehman Optimal Assignment kernel matrix.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, and you have 
    converted these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')

    You can then compute the WL kernel for these graphs:

    >>> K = compute_wl_kernel(grakel_graphs, h=3)

    Note
    ----
    The function requires the GraKeL library to be imported and its WeisfeilerLehmanOptimalAssignment class to 
    be accessible.
    """
    # Initialize the WeisfeilerLehmanOptimalAssignment kernel
    gk = WeisfeilerLehmanOptimalAssignment(n_iter = h, normalize=True)

    # Fit and transform the kernel on the input graph list
    K = gk.fit_transform(grakel_graphs)

    return K

def parse_input(grakel_graphs, h):
    """
    Computes the Weisfeiler-Lehman Optimal Assignment (WL-OA) for feature extraction on a list of GraKeL graphs.

    The WL-OA is a method used to extract feature vectors from graphs. It involves an iterative procedure,
    controlled by the parameter 'h', that aggregates information from a node's neighbors to update its label. 

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects from which to extract features.
    h : int
        The number of iterations for the Weisfeiler-Lehman procedure.

    Returns
    -------
    features : list
        A list of feature vectors, one for each graph in `grakel_graphs`.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, and you have 
    converted these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')

    You can then extract the WL-OA features for these graphs:

    >>> features = parse_input(grakel_graphs, h=3)

    Note
    ----
    The function requires the GraKeL library to be imported and its WeisfeilerLehmanOptimalAssignment class to 
    be accessible.
    """
    # Initialize the WeisfeilerLehmanOptimalAssignment kernel
    gk = WeisfeilerLehmanOptimalAssignment(n_iter = h, normalize=False)

    # Fit and transform the kernel on the input graph list
    gk.fit_transform(grakel_graphs)
    features = gk.parse_input(grakel_graphs)
    return features

def parse_input_vh(grakel_graphs, h):
    """
    Computes the Weisfeiler-Lehman Optimal Assignment (WL-OA) for feature extraction on a list of GraKeL graphs.

    The WL-OA is a method used to extract feature vectors from graphs. It involves an iterative procedure,
    controlled by the parameter 'h', that aggregates information from a node's neighbors to update its label. 

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects from which to extract features.
    h : int
        The number of iterations for the Weisfeiler-Lehman procedure.

    Returns
    -------
    features : list
        A list of feature vectors, one for each graph in `grakel_graphs`.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, and you have 
    converted these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')

    You can then extract the WL-OA features for these graphs:

    >>> features = parse_input(grakel_graphs, h=3)

    Note
    ----
    The function requires the GraKeL library to be imported and its WeisfeilerLehmanOptimalAssignment class to 
    be accessible.
    """
    gk = WeisfeilerLehmanOptimalAssignment(n_iter = h, normalize=False)

    # Fit and transform the kernel on the input graph list
    gk.fit_transform(grakel_graphs)
    features = gk.parse_input(grakel_graphs)
    return features

def representation(grakel_graphs, h):
    """
    Computes the Weisfeiler-Lehman (WL) representation of a list of GraKeL graphs.

    The WL method involves an iterative procedure, controlled by the parameter 'h', that aggregates information 
    from a node's neighbors to update its label. This function returns a dictionary mapping each unique label 
    to a unique integer, effectively providing a 'representation' of the graph substructures in the WL procedure.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects from which to compute the WL representation.
    h : int
        The number of iterations for the Weisfeiler-Lehman procedure.

    Returns
    -------
    _inv_labels : dict
        A dictionary where each unique label in the WL procedure is mapped to a unique integer. The dictionary 
        keys represent substructures in the graphs, and the values are the unique identifiers for these substructures.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, and you have 
    converted these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')

    You can then compute the WL representation for these graphs:

    >>> wl_representation = representation(grakel_graphs, h=1)

    This will output a dictionary where each key represents a unique substructure in the WL-OA (or just WL) procedure and the 
    corresponding value is its unique identifier.

    Note
    ----
    The function requires the GraKeL library to be imported and its WeisfeilerLehman class to be accessible.
    """

    # Initialize the WeisfeilerLehmanOptimalAssignment kernel
    gk = WeisfeilerLehmanOptimalAssignment(n_iter = h, normalize=True)

    # Fit and transform the kernel on the input graph list
    gk.fit_transform(grakel_graphs)
    return gk._inv_labels

def get_classes_from_json(json_file):
    """
    Extracts and returns the classes from a JSON file.

    This function opens the JSON file, loads its content as a dictionary, and iterates through this dictionary 
    to collect the 'class_name' from each item. The classes are then returned as a list.

    Parameters
    ----------
    json_file : str
        The path to the JSON file from which to extract the classes.

    Returns
    -------
    classes : list
        A list of classes extracted from the JSON file.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary where each item has a 'class_name' field:

    >>> classes = get_classes_from_json('data.json')

    This will output a list of the classes found in the JSON file in the same order as parse input function.

    """
    with open(json_file, 'r') as f:
        data_dict = json.load(f)

    classes = []
    for key, val in data_dict.items():
        classes.append(val['class_name'])
        
    return classes

def feynman_SVM(grakel_graphs, class_list, test_size=0.3, cv=5):
    """
    Trains a Support Vector Machine (SVM) classifier on a list of GraKeL graphs and a corresponding list of classes.

    This function first computes a Weisfeiler-Lehman kernel on the input graphs. It then uses this kernel 
    to train a One-vs-One SVM classifier. The function also performs cross-validation on the trained model.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used in the classification.
    class_list : list
        A list of classes corresponding to the input graphs.
    test_size : float, optional
        The proportion of the dataset to include in the test split (default is 0.3).
    cv : int, optional
        The number of cross-validation folds (default is 5).

    Returns
    -------
    clf : OneVsOneClassifier
        A trained SVM classifier with the One-vs-One strategy.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')
    
    You can then train an SVM classifier on these graphs and classes:

    >>> svm_clf = feynman_SVM(grakel_graphs, classes)

    This will output a trained SVM classifier and print the cross-validation accuracy of the model.

    Note
    ----
    The function requires the sklearn library for the SVM classifier and cross-validation, numpy for array 
    manipulation, and it also uses the `compute_wl_kernel` function to compute the Weisfeiler-Lehman kernel.
    """


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
    """
    Trains a Random Forest classifier on a list of GraKeL graphs and a corresponding list of classes.

    This function first calculates the features of the input graphs using the Weisfeiler-Lehman Optimal Assignment
    procedure and takes the values after the first 22 features. It then uses these features to train a Random Forest classifier.
    The function also performs stratified k-fold cross-validation on the trained model.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used in the classification.
    class_list : list
        A list of classes corresponding to the input graphs.

    Returns
    -------
    None

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')
    
    You can then train a Random Forest classifier on these graphs and classes:

    >>> random_forest_classifier(grakel_graphs, classes)

    This will print the cross-validation accuracy of the model.

    Note
    ----
    The function requires the sklearn library for the Random Forest classifier, StratifiedKFold for cross-validation,
    and numpy for array manipulation. It also uses the `parse_input` function to compute the features using 
    the Weisfeiler-Lehman Optimal Assignment procedure.
    """

    # Initialize a Weisfeiler-Lehman subtree kernel
    # Calculate the kernel matrix.
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    # Create numpy array from the classes list
    classes = np.array(class_list)

    rf = RandomForestClassifier(n_estimators=60, max_depth = 2,random_state=42)
    
    # Create StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    scores = []
    for train_index, test_index in skf.split(my_list, classes):
        X_train, X_test = my_list[train_index], my_list[test_index]
        y_train, y_test = classes[train_index], classes[test_index]
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        scores.append(score)
        
    print("Cross-validation accuracy random-forest: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2))


def lime_explainer(grakel_graphs, class_list, index):
    """
    Uses the LIME (Local Interpretable Model-Agnostic Explanations) to explain the predictions of a 
    Random Forest Classifier model trained on graph data.

    LIME is a tool that helps understand the predictions of any classifier in an interpretable and faithful manner.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used in the classification.
    class_list : list
        A list of classes corresponding to the input graphs.
    index : int
        The index of the instance from the list of graphs that we want to explain.

    Returns
    -------
    feature_weights_output : list
        A list of feature weights for each class, explaining the contribution of each feature to the prediction.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')
    
    You can then use LIME to explain the prediction for a particular instance:

    >>> lime_explainer(grakel_graphs, classes, 0)

    This will print out the explanation of the prediction for the instance at index 0, for each class. It will also
    return a list of feature weights for each class.

    Note
    ----
    The function requires the sklearn library for the Random Forest classifier and the LIME library for generating
    explanations. It also uses the `parse_input` function to compute the features using 
    the Weisfeiler-Lehman Optimal Assignment procedure. 
    """
    feature_names_list = ["strong", "weak", "electromagnetic"]
    # Create numpy array from the classes list
    classes = np.array(class_list)
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    rf = RandomForestClassifier(n_estimators=30, max_depth=2,random_state=42)
    rf.fit(my_list, classes)

    explainer = lime.lime_tabular.LimeTabularExplainer(my_list, class_names = feature_names_list, discretize_continuous=True, random_state=42)
    # An instance to justify the model's selection
    i = index
    instance = my_list[i]

    # Generate an explanation
    exp = explainer.explain_instance(instance, rf.predict_proba, num_features=5, top_labels=3)

    # Visualize the explanation for each class
    for i in range(3):
        print('Explanation for class %s:' % i)
        exp.save_to_file(f'explanation_class_{i}.html', labels=[i])  # This saves each explanation to a separate HTML file

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
    """
    Uses the LIME (Local Interpretable Model-Agnostic Explanations) to explain the predictions of a 
    Random Forest Classifier model trained on graph data.

    LIME is a tool that helps understand the predictions of any classifier in an interpretable and faithful manner.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used in the classification.
    class_list : list
        A list of classes corresponding to the input graphs.
    index : int
        The index of the instance from the list of graphs that we want to explain.

    Returns
    -------
    feature_weights_output : list
        A list of feature weights for each class, explaining the contribution of each feature to the prediction.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')
    
    You can then use LIME to explain the prediction for a particular instance:

    >>> lime_explainer(grakel_graphs, classes, 0)

    This will print out the explanation of the prediction for the instance at index 0, for each class. It will also
    return a list of feature weights for each class.

    Note
    ----
    The function requires the sklearn library for the Random Forest classifier and the LIME library for generating
    explanations. It also uses the `parse_input` function to compute the features using 
    the Weisfeiler-Lehman Optimal Assignment procedure. 
    """
    feature_names_list = ["strong", "weak", "electromagnetic"]
    # Create numpy array from the classes list
    classes = np.array(class_list)
    my_list = parse_input_vh(grakel_graphs, 1)
    my_list = [lst[:22] for lst in my_list]
    my_list = np.array(my_list)
    rf = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=42)
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
    """
    Train a Random Forest Classifier on input graph data and plot the error rates of cross-validation folds.

    The function takes as input a list of GraKeL Graph objects and a corresponding list of classes. It trains 
    a Random Forest Classifier on the graphs and performs cross-validation. The error rates for each cross-validation 
    fold are then plotted.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the Random Forest Classifier.
    class_list : list
        A list of classes corresponding to the input graphs.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')

    You can then use this function to visualize the error rates of the cross-validation folds:

    >>> plot_rf_error(grakel_graphs, classes)

    This will create a plot of the error rates for each cross-validation fold.

    Note
    ----
    The function requires the sklearn library for the Random Forest Classifier and cross-validation. 
    It also uses the `parse_input` function to compute the features using 
    the Weisfeiler-Lehman Optimal Assignment procedure. 
    """
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    rf = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=42)
    rf.fit(my_list, classes)

    # Cross-validation of the model
    cv_scores = model_selection.cross_val_score(rf, my_list, classes, cv=8)

    # Calculate the error rates
    errors = [1 - cv for cv in cv_scores]

    # Plot the error rates
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(errors)), errors, marker='o')
    plt.title('Error Rate for each CV fold for Random Forests')
    plt.xlabel('CV fold')
    plt.ylabel('Error rate')
    plt.show()

def plot_knn_error(grakel_graphs, class_list):
    """
    Train a k-Nearest Neighbors Classifier on input graph data and plot the error rates of cross-validation folds.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the k-NN Classifier.
    class_list : list
        A list of classes corresponding to the input graphs.
    """
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(my_list, classes)

    # Cross-validation of the model
    cv_scores = model_selection.cross_val_score(knn, my_list, classes, cv=8)

    # Calculate the error rates
    errors = [1 - cv for cv in cv_scores]

    # Plot the error rates
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(errors)), errors, marker='o')
    plt.title('Error Rate for each CV fold K-Nearest Neighbours')
    plt.xlabel('CV fold')
    plt.ylabel('Error rate')
    plt.show()

def plot_svm_error_scaled(grakel_graphs, class_list):
    """
    Train a Support Vector Machine (SVM) classifier with feature scaling on the input graph data 
    and plot the error rates of cross-validation folds.

    The function takes as input a list of GraKeL Graph objects and a corresponding list of classes. 
    It trains an SVM classifier on the graphs with scaled features and performs cross-validation. 
    The error rates for each cross-validation fold are then plotted.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the SVM classifier.
    class_list : list
        A list of classes corresponding to the input graphs.

    Returns
    -------
    None

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, 
    you have converted these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, 
    and you have a list of classes associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')

    You can then use this function to visualize the error rates of the cross-validation folds:

    >>> plot_svm_error_scaled(grakel_graphs, classes)

    This will create a plot of the error rates for each cross-validation fold.

    Note
    ----
    The function requires the sklearn library for the SVM classifier, cross-validation, 
    and feature scaling. It also uses the `parse_input` function to compute the features 
    using the Weisfeiler-Lehman Optimal Assignment procedure.
    """
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    # Make a pipeline that includes scaling and the SVM classifier
    clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
    clf.fit(my_list, classes)

    # Cross-validation of the model
    cv_scores = model_selection.cross_val_score(clf, my_list, classes, cv=6)

    # Calculate the error rates
    errors = [1 - cv for cv in cv_scores]

    # Plot the error rates
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(errors)), errors, marker='o')
    plt.title('Error Rate for each CV fold Support Vector Machine')
    plt.xlabel('CV fold')
    plt.ylabel('Error rate')
    plt.show()

def plot_error_voting(grakel_graphs, class_list):
    """
    Train a Voting Classifier on input graph data and plot the error rates of cross-validation folds.

    The function takes as input a list of GraKeL Graph objects and a corresponding list of classes. It trains 
    a Voting Classifier on the graphs and performs cross-validation. The error rates for each cross-validation 
    fold are then plotted.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the Random Forest Classifier.
    class_list : list
        A list of classes corresponding to the input graphs.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')

    You can then use this function to visualize the error rates of the cross-validation folds:

    >>> plot_rf_error(grakel_graphs, classes)

    This will create a plot of the error rates for each cross-validation fold.

    Note
    ----
    The function requires the sklearn library for the Random Forest Classifier and cross-validation. 
    It also uses the `parse_input` function to compute the features using 
    the Weisfeiler-Lehman Optimal Assignment procedure. 
    """

    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    # Define the individual classifiers
    clf1 = RandomForestClassifier(n_estimators=40, max_depth=2, random_state=42)
    clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    voting_clf = VotingClassifier(estimators=[('rf', clf1), ('svc', clf2), ('knn', knn_clf)], voting='hard')

    # Cross-validation of the model
    cv_scores = model_selection.cross_val_score(voting_clf, my_list, classes, cv=8)

    # Calculate the error rates
    errors = [1 - cv for cv in cv_scores]

    # Plot the error rates
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(errors)), errors, marker='o')
    plt.title('Error Rate for each CV fold for Voting Classifier')
    plt.xlabel('CV fold')
    plt.ylabel('Error rate')
    plt.show()



def plot_errors(grakel_graphs, class_list):
    """
    Train a Random Forest, k-Nearest Neighbors, and Support Vector Machine classifiers 
    on input graph data and plot the error rates of cross-validation folds for each classifier.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the classifiers.
    class_list : list
        A list of classes corresponding to the input graphs.
    """
    # Plot the error rate for Random Forest classifier
    print("Error rate for Random Forest Classifier:")
    plot_rf_error(grakel_graphs, class_list)

    # Plot the error rate for k-Nearest Neighbors classifier
    print("Error rate for k-Nearest Neighbors Classifier:")
    plot_knn_error(grakel_graphs, class_list)

    # Plot the error rate for Support Vector Machine classifier
    print("Error rate for Support Vector Machine Classifier:")
    plot_svm_error_scaled(grakel_graphs, class_list)

    # Plot the error rate for Voting Classifier
    print("Error rate for Voting Classifier:")
    plot_error_voting(grakel_graphs, class_list)

def compare_classifiers(grakel_graphs, class_list):
    """
    Train four different classifiers on input graph data and plot their respective error rates on the same plot.

    The function takes as input a list of GraKeL Graph objects and a corresponding list of classes. It trains 
    four classifiers (K-Nearest Neighbors, Scaled SVM, Random Forest, and Voting Classifier) on the graphs and 
    performs cross-validation for each. The error rates for each cross-validation fold are then plotted for 
    all classifiers in the same figure.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the classifiers.
    class_list : list
        A list of classes corresponding to the input graphs.

    Returns
    -------
    None

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, 
    you have converted these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, 
    and you have a list of classes associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')

    You can then use this function to compare the performance of four classifiers:

    >>> compare_classifiers(grakel_graphs, classes)

    This will create a plot of the error rates for each cross-validation fold for all classifiers.

    Note
    ----
    The function requires the sklearn library for the classifiers, cross-validation, and feature scaling. 
    It also uses the `parse_input` function to compute the features using the 
    Weisfeiler-Lehman Optimal Assignment procedure.
    """
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    # Define the individual classifiers
    clf1 = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42)
    clf2 = make_pipeline(StandardScaler(), svm.SVC(C=0.1 , gamma='auto', probability=True))
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    voting_clf = VotingClassifier(estimators=[('rf', clf1), ('svc', clf2), ('knn', knn_clf)], voting='soft')

    # List of classifiers
    classifiers = [knn_clf, clf2, clf1, voting_clf]

    # Classifier names
    classifier_names = ['K-Nearest Neighbors', 'Scaled SVM', 'Random Forest', 'Voting Classifier']

    # Initialize the plot
    plt.figure(figsize=(10, 5))

    # Loop through classifiers
    for clf, name in zip(classifiers, classifier_names):
        # Fit classifier
        clf.fit(my_list, classes)

        # Cross-validation of the model
        cv_scores = model_selection.cross_val_score(clf, my_list, classes, cv=8)

        # Calculate the error rates
        errors = [1 - cv for cv in cv_scores]

        # Plot the error rates
        plt.plot(range(len(errors)), errors, marker='o', label=name)

    plt.title('Error Rate for each CV fold')
    plt.xlabel('CV fold')
    plt.ylabel('Error rate')
    plt.legend(loc='upper right')
    plt.show()




def plot_confusion_matrix(grakel_graphs, class_list):
    """
    Given a list of Grakel graphs and corresponding class labels, the function performs a train-test split,
    trains a Random Forest classifier, predicts labels for the test set and plots a confusion matrix.

    Parameters
    ----------
    grakel_graphs : list
        A list of Grakel Graph objects to be used for training the Random Forest classifier.
    class_list : list
        A list of classes corresponding to the input graphs.
    """

    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[:22] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(my_list, classes, test_size=0.25, random_state=42)

    rf = RandomForestClassifier(n_estimators=40, max_depth=2, random_state=42)
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

def plot_confusion_matrix_voting(grakel_graphs, class_list):

    """
    Given a list of Grakel graphs and corresponding class labels, the function performs a train-test split,
    trains a Random Forest classifier, predicts labels for the test set and plots a confusion matrix.

    Parameters
    ----------
    grakel_graphs : list
        A list of Grakel Graph objects to be used for training the Random Forest classifier.
    class_list : list
        A list of classes corresponding to the input graphs.
    """

    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[:22] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(my_list, classes, test_size=0.2, random_state=42)

    # Define the individual classifiers
    # Define the individual classifiers
    rnd_clf = RandomForestClassifier(n_estimators=40, max_depth=2, random_state=42)
    svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    # log_clf = LogisticRegression(solver='liblinear', random_state=42)
    knn_clf = KNeighborsClassifier(n_neighbors=3)

    # Combine them into a voting classifier
    voting_clf = VotingClassifier(estimators=[('rf', rnd_clf), ('svc', svm_clf), ('knn', knn_clf)], voting='soft')
    eclf = voting_clf.fit(X_train, y_train)

    y_pred = eclf.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



def find_substructure(graph, labels):
    """
    Finds the subgraph within the provided graph that contains the specified node labels and is connected.

    This function will iterate through all possible combinations of nodes within the provided graph to find a
    connected subgraph that contains the exact labels provided. The labels are sorted for comparison purposes.

    Parameters
    ----------
    graph : NetworkX Graph
        The graph within which to find the subgraph.
    labels : list
        A sorted list of node labels to find within the graph.

    Returns
    -------
    tuple (set, set)
        A tuple containing two sets. The first set contains the nodes of the subgraph. The second set contains the 
        edges of the subgraph. If no suitable subgraph is found, it returns two empty sets.

    Examples
    --------
    Suppose you have a NetworkX graph 'G' and a list of node labels:

    >>> G = nx.Graph()
    >>> G.add_edge('A', 'B', label='1')
    >>> G.add_edge('B', 'C', label='2')
    >>> G.add_edge('C', 'D', label='3')
    >>> labels = ['A', 'B', 'C']

    You can then use this function to find the subgraph with the specified labels:

    >>> nodes, edges = find_substructure(G, labels)

    This will return the nodes and edges of the subgraph containing the nodes 'A', 'B', and 'C'.

    Note
    ----
    This function uses the NetworkX package for graph manipulation.
    """
    labels.sort()

    for combination in combinations(graph.nodes(data=True), len(labels)):
        nodes, data = zip(*combination)
        node_labels = sorted([d['label'] for d in data])
        
        # Check if labels match
        if labels == node_labels:
            subgraph = graph.subgraph(nodes)

            # Check if subgraph is connected
            if nx.is_connected(subgraph):
                subgraph_edges = list(subgraph.edges())
                return nodes, subgraph_edges

    # If no suitable subgraph found
    return set(), set()



def visualize_substructure(graph, subgraph_nodes, subgraph_edges):
    """
    Visualize a given graph with a highlighted subgraph defined by its nodes and edges.

    The function plots the graph using NetworkX's drawing utilities, highlighting the provided subgraph nodes 
    and edges in a different color.

    Parameters
    ----------
    graph : NetworkX Graph
        The graph to be visualized.
    subgraph_nodes : set
        The nodes of the subgraph to be highlighted.
    subgraph_edges : set
        The edges of the subgraph to be highlighted.

    Examples
    --------
    Suppose you have a NetworkX graph 'G', and you have determined a subgraph of interest:

    >>> G = nx.Graph()
    >>> G.add_edge('A', 'B', label='1')
    >>> G.add_edge('B', 'C', label='2')
    >>> G.add_edge('C', 'D', label='3')
    >>> nodes = {'A', 'B', 'C'}
    >>> edges = {('A', 'B'), ('B', 'C')}

    You can then use this function to visualize the graph with the subgraph highlighted:

    >>> visualize_substructure(G, nodes, edges)

    This will create a plot of the graph 'G' with nodes 'A', 'B', and 'C' and edges ('A', 'B') and ('B', 'C') 
    highlighted.

    Note
    ----
    This function uses the NetworkX and Matplotlib libraries for graph visualization.
    """

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
    """
    Convert a GraKeL Graph object to a NetworkX Graph object.

    The function takes as input a GraKeL Graph object, extracts its nodes, node labels, and edges, 
    and constructs a corresponding NetworkX graph. It is useful when you want to use the NetworkX 
    library's functionalities on a graph that is originally in GraKeL format.

    Parameters
    ----------
    grakel_graph : GraKeL Graph
        The GraKeL Graph object to be converted.

    Returns
    -------
    networkx_graph : NetworkX Graph
        The converted NetworkX graph.

    Examples
    --------
    Suppose you have a GraKeL graph 'g':

    >>> g = grakel.graph_from_networkx(nx.complete_graph(5))

    You can then use this function to convert it to a NetworkX graph:

    >>> nx_graph = grakel_to_networkx(g)

    You can then use the various functionalities provided by the NetworkX library on 'nx_graph'.

    Note
    ----
    This function uses the NetworkX and GraKeL libraries. Make sure to install these libraries before using 
    the function.
    """

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
    """
    Extract the labels of substructures (e.g., subgraphs) from a feature vector.

    The function takes as input a feature vector, which represents a graph in the form of a list 
    of binary features, each indicating the presence or absence of a particular substructure 
    in the graph. It then extracts the labels of the substructures corresponding to the non-zero 
    entries in the feature vector.

    Parameters
    ----------
    feature_vector : list
        A list of binary features representing a graph.

    Returns
    -------
    substructure_labels : list
        A list of lists, where each sublist contains the labels of the nodes that form a 
        particular substructure in the graph.

    Examples
    --------
    Suppose you have a feature vector 'v' that represents a graph:

    >>> v = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    You can then use this function to extract the labels of the substructures represented by 'v':

    >>> labels = get_substructure_labels(v)

    Note
    ----
    The function is based on the assumption that each position in the feature vector corresponds to a 
    unique substructure, and that a position's index (plus 22) corresponds to its key in the feature map.
    """

    feature_map = representation(graphs, 1)[1]
    node_mappings = representation(graphs, 1)[0]
    inverse_node_mappings = {v: k for k, v in node_mappings.items()}
    # Get indices of non-zero entries in the feature vector, increment each index by 22
    positions = [index+22 for index, value in enumerate(feature_vector) if value > 0]
    
    # Translate positions to the corresponding keys in the feature map
    keys = [k for k, v in feature_map.items() if v in positions]
    
    # Extract the node labels from the keys and translate them back to the original labels
    substructure_labels = []
    for pos, key in zip(positions, keys):
        # extract all numbers from the key and convert them to integers
        node_indices = [int(s) for s in findall(r'\b\d+\b', key)]
        labels = [inverse_node_mappings[i] for i in node_indices]
        substructure_labels.append((pos-22, labels))  # Subtract 22 before displaying

    return substructure_labels

def process_graph(index, json_file):
    """
    Load a graph from a JSON file, extract its substructures, and visualize them.

    The function takes as input an index and a JSON file. The JSON file should contain a dictionary 
    where the keys are graph names and the values are dictionaries containing the graph data. The 
    index should correspond to the position of the graph in the JSON file. The function will then 
    parse the input, extract the feature vector for the specified graph, and get the labels of 
    the substructures. It will then load the graph data, convert it to a NetworkX graph, and find 
    and visualize the substructures.

    Parameters
    ----------
    index : int
        The position of the graph in the JSON file.
    json_file : str
        The path to the JSON file containing the graph data.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs:

    >>> process_graph(0, 'data.json')

    This will load the first graph in the JSON file, extract its substructures, and visualize them.

    Note
    ----
    The function requires the NetworkX library for graph operations and the Matplotlib library for 
    visualization. It also uses the `parse_input`, `get_substructure_labels`, `networkx_obj_to_grakel`, 
    `grakel_to_networkx`, and `find_substructure` functions, which should be defined in the same scope.
    """

    # Parse the input and get the feature vector
    my_list = parse_input(graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    feature_vector = np.array(my_list[index])
    
    # Get the substructure labels
    substructure_labels = get_substructure_labels(feature_vector)

    # Load the graph data
    with open(json_file, 'r') as f:
        grakel_graphs = json.load(f)
    
    # Get the graph name from the keys of the dictionary using the index
    graph_name = list(grakel_graphs.keys())[index]
    graph = grakel_graphs[graph_name]["networkx_obj"]

    # Convert to GraKeL graph
    graph = networkx_obj_to_grakel(graph)
    
    # Convert to NetworkX graph
    g_nx = grakel_to_networkx(graph)
    
    # Define the number of columns for subplot
    ncols = 3
    
    # Calculate the number of rows needed for the subplot
    nrows = -(-len(substructure_labels) // ncols)  # Ceiling division

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 10 * nrows))

    # Ensure axs is always a 2D array
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs[np.newaxis, :]
    elif ncols == 1:
        axs = axs[:, np.newaxis]

    for i, (feature_number, labels) in enumerate(substructure_labels):
        # Calculate the row and column indices
        row = i // ncols
        col = i % ncols

        # Find the substructure
        subgraph_nodes, subgraph_edges = find_substructure(g_nx, labels)

        # Create position layout
        pos = nx.spring_layout(g_nx)

        # Draw all nodes and edges in the graph in light color
        nx.draw_networkx(g_nx, pos, node_color='blue', edge_color='black', ax=axs[row, col])

        # Draw the nodes and edges of the subgraph in dark color
        nx.draw_networkx_nodes(g_nx, pos, nodelist=subgraph_nodes, node_color='red', ax=axs[row, col])
        nx.draw_networkx_edges(g_nx, pos, edgelist=subgraph_edges, edge_color='red', ax=axs[row, col])
        nx.draw_networkx_labels(g_nx, pos, ax=axs[row, col])

        # Print the feature number and labels below the graph
        axs[row, col].set_title(f'Feature number: {feature_number}, Labels: {labels}', fontsize=23.5)

    # Remove unused subplots
    for i in range(len(substructure_labels), nrows*ncols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.show()

def grid_search_voting_classifier(grakel_graphs, class_list):
    """
    Performs a grid search to tune the parameters of a voting classifier built on a Random Forest, an SVM, 
    and a K-Nearest Neighbors model.
    
    The function takes as input a list of GraKeL Graph objects and a corresponding list of classes. It trains 
    a voting classifier and performs a grid search to find the optimal parameters. 

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the classifier.
    class_list : list
        A list of classes corresponding to the input graphs.

    Returns
    -------
    dict
        The best parameters found by the grid search.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')

    You can then use this function to find the best parameters for the voting classifier:

    >>> best_params = grid_search_voting_classifier(grakel_graphs, classes)
    >>> print(best_params)

    This will output the optimal parameters found by the grid search.

    Note
    ----
    The function requires the sklearn library for the classifiers and the grid search. 
    It also uses the `parse_input` function to compute the features using 
    the Weisfeiler-Lehman Optimal Assignment procedure. 
    """
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    # Define the individual classifiers
    clf1 = RandomForestClassifier(random_state=42)
    clf2 = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', probability=True))
    knn_clf = KNeighborsClassifier()

    # Define the Voting Classifier
    voting_clf = VotingClassifier(estimators=[('rf', clf1), ('svc', clf2), ('knn', knn_clf)], voting='soft')

    # Define the parameter grid for the grid search
    param_grid = {
        'rf__n_estimators': [10, 20, 30, 40],
        'rf__max_depth': [2, 5, 10, None],
        'svc__svc__C': [0.1, 1, 10],
        'knn__n_neighbors': [3, 5, 7, 10]
    }

    # Define the grid search
    grid_search = model_selection.GridSearchCV(voting_clf, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search
    grid_result = grid_search.fit(my_list, classes)

    # Return the best parameters found by the grid search
    return grid_result.best_params_




def plot_hyperparameter_tuning_rf(grakel_graphs, class_list, depth_range, trees_range):
    # Calculate the kernel matrix.
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[:22] for lst in my_list]
    my_list = np.array(my_list)

    # Create numpy array from the classes list
    classes = np.array(class_list)

    # Define colors for different depths
    colors = ['red', 'green', 'blue', 'yellow', 'magenta']

    max_cv_score = 0
    best_params = None

    # For each depth
    for i, depth in enumerate(depth_range):
        cv_scores = []
        # For each number of trees
        for trees in trees_range:
            rf = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=42)
            scores = model_selection.cross_val_score(rf, my_list, classes, cv=5)

            avg_cv_score = scores.mean()
            cv_scores.append(avg_cv_score)
            if avg_cv_score > max_cv_score:
                max_cv_score = avg_cv_score
                best_params = (depth, trees)

        # Smooth the curve
        xnew = np.linspace(min(trees_range), max(trees_range), 300) 
        spl = make_interp_spline(trees_range, cv_scores, k=3)  # type: BSpline
        cv_scores_smooth = spl(xnew)

        # Plot the results
        plt.plot(xnew, cv_scores_smooth, color=colors[i], label=f'depth={depth}')

    plt.legend(loc='best')
    plt.xlabel('Number of Trees')
    plt.ylabel('Cross-Validation Score')
    plt.title('Hyperparameter tuning of Random Forest')
    plt.grid(True)
    plt.show()

    print(f'The best parameters are depth={best_params[0]}, trees={best_params[1]} with a cross-validation score of {max_cv_score:.2f}')

def lime_explainer_voting(grakel_graphs, class_list, index):
    """
    Generates explanations for the predictions of a voting classifier using LIME (Local Interpretable Model-Agnostic 
    Explanations). The voting classifier is comprised of a Random Forest Classifier and a Support Vector Classifier.

    Args:
        grakel_graphs (list): List of Grakel Graphs.
        class_list (list): The target classes for the instances.
        index (int): Index of the instance to explain.

    Returns:
        list: A list of feature importance weights for each class. Each item in the list is a list containing a 
              tuple, where the first element is the feature name and the second element is the weight of the feature.

    Notes:
        This function also saves the explanations for each class to separate HTML files named 'explanation_class_#.html',
        where '#' is replaced by the class index.
    """
    feature_names_list = ["strong", "weak", "electromagnetic"]
    # Create numpy array from the classes list
    classes = np.array(class_list)
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)

    # Define the individual classifiers
    clf1 = RandomForestClassifier(n_estimators=40, max_depth=5,random_state=42)
    clf2 = make_pipeline(StandardScaler(), SVC(C=0.1, gamma='auto', probability=True))
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    voting_clf = VotingClassifier(estimators=[('rf', clf1), ('svc', clf2), ('knn', knn_clf)], voting='soft')

    # Combine them into a voting classifier
    eclf = voting_clf.fit(my_list, classes)

    explainer = lime.lime_tabular.LimeTabularExplainer(my_list, class_names = feature_names_list, discretize_continuous=True, random_state=42)

    # An instance to justify the model's selection
    i = index
    instance = my_list[i]

    # Generate an explanation
    exp = explainer.explain_instance(instance, eclf.predict_proba, num_features=20, top_labels=3)

    # Visualize the explanation for each class
    for i in range(3):
        print('Explanation for class %s:' % i)
        exp.save_to_file(f'explanation_class_{i}.html', labels=[i])  # This saves each explanation to a separate HTML file

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


def fit_vote_classify(grakel_graphs, classes):
    """
    Trains and evaluates individual and voting classifiers on input graph data.

    The function takes as input a list of GraKeL Graph objects and a corresponding list of classes. It trains 
    individual classifiers (Random Forest, SVM, and K-Nearest Neighbors) as well as a voting classifier on the data. 
    It also calculates and outputs the cross-validation scores for each classifier.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the classifiers.
    classes : list
        A list of classes corresponding to the input graphs.

    Returns
    -------
    None
        The function doesn't return anything but prints the cross-validation scores for each classifier.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')

    You can then use this function to train and evaluate the classifiers:

    >>> fit_vote_classify(grakel_graphs, classes)

    This will print the cross-validation scores for each classifier.

    Note
    ----
    The function requires the sklearn library for the classifiers and the cross-validation. 
    It also uses the `parse_input` function to compute the features using 
    the Weisfeiler-Lehman Optimal Assignment procedure. 
    """

    # Create numpy array from the classes list
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    classes = np.array(classes)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(my_list, classes, test_size=0.3, random_state=42)

    # Define the individual classifiers
    rnd_clf = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42)
    svm_clf = make_pipeline(StandardScaler(), SVC(C=1, gamma='auto', probability=True))
    # log_clf = LogisticRegression(solver='liblinear', random_state=42)
    knn_clf = KNeighborsClassifier(n_neighbors=3)

    # Combine them into a voting classifier
    voting_clf = VotingClassifier(estimators=[('rf', rnd_clf), ('svc', svm_clf), ('knn', knn_clf)], voting='soft')

    # Fit the classifiers and print their accuracy and cross-validation scores
    for clf in (rnd_clf, svm_clf, knn_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("\n"+clf.__class__.__name__)
        
        # Calculate cross-validation score
        scores = cross_val_score(clf, my_list, classes, cv=8)
        print("Cross-validation score: ", scores.mean())

def plot_pca_transform(grakel_graphs, class_list):
    """
    Performs Principal Component Analysis (PCA) on the input graph data and visualizes the data in 2D.

    PCA is a dimensionality reduction technique that can be useful in visualizing high-dimensional data. 

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for the PCA transformation.
    class_list : list
        A list of classes corresponding to the input graphs.

    Examples
    --------
    Suppose you have a JSON file 'data.json' that contains a dictionary with NetworkX graphs, you have converted 
    these into GraKeL graphs using the `convert_dict_to_grakel_graphs` function, and you have a list of classes 
    associated with these graphs:

    >>> grakel_graphs = convert_dict_to_grakel_graphs('data.json')
    >>> classes = get_classes_from_json('data.json')

    You can then use this function to visualize the PCA-transformed data:

    >>> plot_pca_transform(grakel_graphs, classes)

    This will create a 2D scatter plot where the color of the points indicates the class of the instance.

    Note
    ----
    The function requires the sklearn library for the PCA transformation and matplotlib for the visualization. 
    It also uses the `parse_input` function to compute the features using 
    the Weisfeiler-Lehman Optimal Assignment procedure. 
    """
    # Convert labels to numerical values
    classes_encoded = list(map(int, class_list))
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    # Perform PCA
    pca = PCA(n_components=3)
    transformed_data = pca.fit_transform(my_list)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=classes_encoded, alpha=0.6, edgecolors='w')

    # Add a color bar
    plt.colorbar(scatter)

    # Create a mapping from class labels to class names
    class_names = {0: "strong", 1: "weak", 2: "EM"}

    # Create legend handles
    class_legend = [mpatches.Patch(color=plt.cm.viridis(i/2), label=name) for i, name in class_names.items()]

    # Add legend to the plot
    plt.legend(handles=class_legend, loc='upper right')

    plt.show()

def rf_performance_depth(grakel_graphs, classes, n_estimators_values, max_depth_values):
    """
    Trains a Random Forest classifier for each combination of n_estimators and max_depth and plots the resulting cross-validation scores.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the classifiers.
    classes : list
        A list of classes corresponding to the input graphs.
    n_estimators_values : list
        A list of n_estimators values for which to train the Random Forest classifier and calculate cross-validation scores.
    max_depth_values : list
        A list of max_depth values for which to train the Random Forest classifier and calculate cross-validation scores.

    Returns
    -------
    None
    """
    
    # Convert the input to numpy arrays
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    classes = np.array(classes)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(my_list, classes, test_size=0.3, random_state=42)

    # Prepare the plot
    plt.figure(figsize=(12,8))

    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            # Train Random Forest with each n_estimators and max_depth value and calculate cross-validation score
            rnd_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rnd_clf.fit(X_train, y_train)
            
            # Calculate cross-validation scores and then convert them into errors
            scores = cross_val_score(rnd_clf, my_list, classes, cv=8)
            errors = [1 - score for score in scores]
            
            # Plot cross-validation error for current n_estimators and max_depth values
            plt.plot(range(1, 9), errors, marker='o', label=f'n_estimators = {n_estimators}, max_depth = {max_depth}')

    plt.title('Cross-Validation Errors for Different n_estimators and max_depth Values')
    plt.xlabel('Fold')
    plt.ylabel('Cross-validation error')
    plt.legend()
    plt.show()

def knn_performance(grakel_graphs, classes, n_neighbors_values):
    """
    Trains a k-Nearest Neighbors classifier for each value of n_neighbors and plots the resulting cross-validation scores.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the classifiers.
    classes : list
        A list of classes corresponding to the input graphs.
    n_neighbors_values : list
        A list of n_neighbors values for which to train the k-Nearest Neighbors classifier and calculate cross-validation scores.

    Returns
    -------
    None
    """
    
    # Convert the input to numpy arrays
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    classes = np.array(classes)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(my_list, classes, test_size=0.3, random_state=42)

    # Prepare the plot
    plt.figure(figsize=(10,6))

    for n_neighbors in n_neighbors_values:
        # Train k-Nearest Neighbors with each n_neighbors value and calculate cross-validation score
        knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_clf.fit(X_train, y_train)
        
        # Calculate cross-validation scores and then convert them into errors
        scores = cross_val_score(knn_clf, my_list, classes, cv=8)
        errors = [1 - score for score in scores]
        
        # Plot cross-validation error for current n_neighbors value
        plt.plot(range(1, 9), errors, marker='o', label=f'n_neighbors = {n_neighbors}')

    plt.title('Cross-Validation Errors for Different n_neighbors Values')
    plt.xlabel('Fold')
    plt.ylabel('Cross-validation error')
    plt.legend()
    plt.show()


def svm_performance(grakel_graphs, classes, C_values):
    """
    Trains an SVM classifier for each value of C and plots the resulting cross-validation scores.

    Parameters
    ----------
    grakel_graphs : list
        A list of GraKeL Graph objects to be used for training the classifiers.
    classes : list
        A list of classes corresponding to the input graphs.
    C_values : list
        A list of C values for which to train the SVM classifier and calculate cross-validation scores.

    Returns
    -------
    None
    """
    
    # Create numpy array from the classes list
    my_list = parse_input(grakel_graphs, 1)
    my_list = [lst[22:] for lst in my_list]
    my_list = np.array(my_list)
    classes = np.array(classes)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(my_list, classes, test_size=0.3, random_state=42)
    
    # Prepare the plot
    plt.figure(figsize=(10,6))

    for C_value in C_values:
        # Train SVM with each C value and calculate cross-validation score
        svm_clf = make_pipeline(StandardScaler(), SVC(C=C_value, gamma='auto', probability=True))
        svm_clf.fit(X_train, y_train)
        
        # We calculate cross-validation scores and then convert them into errors
        scores = cross_val_score(svm_clf, my_list, classes, cv=8)
        errors = [1 - score for score in scores]
        
        # Plot cross-validation error for current C value
        plt.plot(range(1, 9), errors, marker='o', label=f'C = {C_value}')

    plt.title('Cross-Validation Errors for Different C Values')
    plt.xlabel('CV-Fold')
    plt.ylabel('Cross-validation error')
    plt.legend()
    plt.show()



json_file = 'dictionary.json'
classes = get_classes_from_json(json_file)
graphs = convert_dict_to_grakel_graphs("dictionary.json")
depth_range = range(2, 7) 
trees_range = range(10, 200, 10) 















































































































































