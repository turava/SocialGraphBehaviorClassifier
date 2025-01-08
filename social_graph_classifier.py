import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

def construct_graph(interactions):
    logging.info("Constructing the interaction graph.")
    G = nx.DiGraph()
    for user_from, user_to, interaction_type in interactions:
        if not G.has_edge(user_from, user_to):
            G.add_edge(user_from, user_to, weight=0)
        G[user_from][user_to]['weight'] += 1
    logging.info("Graph constructed with %d nodes and %d edges.", len(G.nodes), len(G.edges))
    return G

def extract_features(G):
    logging.info("Extracting features from the graph.")
    features = []
    for node in G.nodes:
        in_degree = G.in_degree(node, weight='weight')
        out_degree = G.out_degree(node, weight='weight')
        total_degree = in_degree + out_degree
        features.append({
            'user_id': node,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': total_degree
        })
    logging.info("Features extracted for %d users.", len(features))
    return pd.DataFrame(features)

def create_dataset(features, user_labels):
    logging.info("Creating the dataset by merging features with user labels.")
    dataset = features.merge(user_labels, on='user_id')
    logging.info("Dataset created with %d entries.", len(dataset))
    return dataset

def classify_user(G, user_id, model):
    logging.info("Classifying user: %s", user_id)
    if user_id not in G:
        logging.warning("User %s not found in the graph. Defaulting to 'inactive'.", user_id)
        return "inactive"

    user_features = extract_features(G).set_index('user_id').loc[user_id].values.reshape(1, -1)
    classification = model.predict(user_features)[0]
    logging.info("User %s classified as %s.", user_id, classification)
    return classification

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the classification pipeline.")

    interactions = [
        ("user1", "user2", "like"),
        ("user2", "user3", "comment"),
        ("user3", "user1", "follow"),
        ("user1", "user3", "like"),
        ("user4", "user2", "follow"),
    ]

    user_labels = pd.DataFrame({
        'user_id': ["user1", "user2", "user3", "user4"],
        'label': ["influential", "average", "average", "inactive"]
    })

    G = construct_graph(interactions)
    features = extract_features(G)
    dataset = create_dataset(features, user_labels)

    X = dataset.drop(columns=['user_id', 'label'])
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Training the Random Forest model.")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    logging.info("Evaluating the model.")
    y_pred = model.predict(X_test)
    logging.info("Model evaluation results:\n%s", classification_report(y_test, y_pred))

    new_user_id = "user1"
    classification = classify_user(G, new_user_id, model)
    logging.info("Final classification for user %s: %s", new_user_id, classification)

if __name__ == "__main__":
    main()
