import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

filename = "dictionary.json"
if os.path.exists(filename):
    # Load the existing data from the JSON file
    with open(filename, "r") as file:
        existing_data = json.load(file)
else:
    existing_data = {}

dictionary = {}
terminated = False
name = ""
image = ""
networkx = None
class_label = ""

while not terminated:
    print("Enter 'object name' to enter object name (dictionary key).")
    print("Enter 'networkx' for object creation.")
    print("Enter 'image' to enter image name (ending in .png etc).")
    print("Enter 'class label' (type of majority interaction) to perform majority interaction.")
    print("Enter 'return' to return to the menu.")
    print("Enter 'quit' to quit.")

    prompt = input("Please enter your choice: ")

    if prompt == "quit":
        terminated = True
    elif prompt == "object name":
        name = input("Enter the object name: ")
    elif prompt == "networkx":
        # Creating a NetworkX object
        G = nx.Graph()

        print("NetworkX Object Creation: ")
        print("Please enter the nodes and edges for the graph:")
        print("Enter 'return' to finish creating the graph.")

        while True:
            node = input("Enter a node (or 'return'): ")
            if node == "return":
                break
            G.add_node(node)

        while True:
            edge_start = input("Enter the starting node of an edge (or 'return'): ")
            if edge_start == "return":
                break
            edge_end = input("Enter the ending node of the edge: ")
            G.add_edge(edge_start, edge_end)

        networkx = G
    elif prompt == "image":
        image = input("Enter the image name: ")
    elif prompt == "class label":
        class_label = input("Enter the class label, (0 for strong, 1 for weak, 2 for EM): ")
    elif prompt == "return":
        continue

    # Convert NetworkX object to JSON-compatible format
    if networkx is not None:
        networkx_json = nx.readwrite.json_graph.node_link_data(networkx)
    else:
        networkx_json = None
    # Add the values to the dictionary if name is not empty
    if name != "":
        dictionary[name] = {
            "image": image,
            "networkx_obj": networkx_json,
            "class_name": class_label
        }

# Update the existing data with the new dictionary entries
existing_data.update(dictionary)

# Save the updated data to the JSON file
with open(filename, "w") as file:
    json.dump(existing_data, file)

print(f"Dictionary appended to {filename}")