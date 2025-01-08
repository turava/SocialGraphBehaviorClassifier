import unittest
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from social_graph_classifier import construct_graph, extract_features, create_dataset, classify_user

class TestSocialGraphBehaviorClassifier(unittest.TestCase):

    def setUp(self):
        # Example interactions and labels for testing
        self.interactions = [
            ("user1", "user2", "like"),
            ("user2", "user3", "comment"),
            ("user3", "user1", "follow"),
            ("user1", "user3", "like"),
            ("user4", "user2", "follow"),
        ]

        self.user_labels = pd.DataFrame({
            'user_id': ["user1", "user2", "user3", "user4"],
            'label': ["influential", "average", "average", "inactive"]
        })

        # Create graph and dataset
        self.graph = construct_graph(self.interactions)
        self.features = extract_features(self.graph)
        self.dataset = create_dataset(self.features, self.user_labels)

        # Train model for testing
        X = self.dataset.drop(columns=['user_id', 'label'])
        y = self.dataset['label']
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X, y)

    def test_construct_graph(self):
        # Check that the graph is constructed correctly
        self.assertEqual(len(self.graph.nodes), 4)
        self.assertEqual(len(self.graph.edges), 5)
        self.assertEqual(self.graph["user1"]["user2"]["weight"], 1)

    def test_extract_features(self):
        # Check feature extraction
        self.assertIn('in_degree', self.features.columns)
        self.assertIn('out_degree', self.features.columns)
        self.assertIn('total_degree', self.features.columns)
        self.assertEqual(len(self.features), 4)

    def test_create_dataset(self):
        # Check dataset creation
        self.assertEqual(len(self.dataset), 4)
        self.assertIn('in_degree', self.dataset.columns)
        self.assertIn('label', self.dataset.columns)

    def test_classify_user_existing(self):
        # Test classification for an existing user
        classification = classify_user(self.graph, "user1", self.model)
        self.assertIn(classification, ["inactive", "average", "influential"])

    def test_classify_user_nonexistent(self):
        # Test classification for a non-existent user
        classification = classify_user(self.graph, "nonexistent_user", self.model)
        self.assertEqual(classification, "inactive")

if __name__ == "__main__":
    unittest.main()
