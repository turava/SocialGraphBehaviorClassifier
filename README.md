# SocialGraphBehaviorClassifier
A machine learning pipeline to classify user behaviors in a social network using directed graphs, distinguishing between inactive, average, and influential users.

1. **Inactive Users**: Users with minimal or no interactions on the platform.
2. **Average Users**: Users who interact regularly.
3. **Influential Users**: Users with high levels of interaction and significant influence.

The solution uses a directed graph to model user interactions and machine learning to classify behaviors based on extracted features.

## Features

- **Graph Representation**: User interactions are modeled as a directed graph where:
  - Nodes represent users.
  - Directed edges represent interactions (e.g., follows, likes, comments).
  - Weights on edges indicate the frequency or intensity of interactions.
- **Feature Extraction**: Node-based metrics (e.g., in-degree, out-degree) are calculated to quantify user behavior.
- **Machine Learning Classification**: A Random Forest Classifier is used to categorize users based on their extracted features.
- **Logging**: Detailed logging is implemented to track pipeline stages and results.

## Installation

### Prerequisites
Ensure Python 3.6 or later is installed. You also need the following Python libraries:

- `networkx`
- `pandas`
- `numpy`
- `scikit-learn`

### Setup
Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Interaction Data
Prepare interaction data in the following format:

```python
interactions = [
    ("user1", "user2", "like"),
    ("user2", "user3", "comment"),
    ("user3", "user1", "follow"),
    ("user1", "user3", "like"),
    ("user4", "user2", "follow"),
]
```

Each tuple represents an interaction where:
- The first element is the initiating user.
- The second element is the target user.
- The third element is the type of interaction.

### 2. User Labels
Define user categories as:

```python
user_labels = pd.DataFrame({
    'user_id': ["user1", "user2", "user3", "user4"],
    'label': ["influential", "average", "average", "inactive"]
})
```

### 3. Run the Script
Run the Python script to execute the pipeline:

```bash
python3 social_graph_classifier.py
```

### 4. Results
The script outputs:
- Classification performance metrics (precision, recall, F1-score).
- Classification for a specified user.

## Key Functions

### `construct_graph(interactions)`
- Creates a directed graph from the interaction data.
- Adds weighted edges to represent interaction intensity.

### `extract_features(G)`
- Extracts node-level features from the graph, including:
  - `in_degree`: Number of followers (weighted by interaction frequency).
  - `out_degree`: Number of interactions initiated.
  - `total_degree`: Sum of in-degree and out-degree.

### `create_dataset(features, user_labels)`
- Merges extracted features with user labels to create a dataset for model training.

### `classify_user(G, user_id, model)`
- Classifies a user based on their graph-derived features.
- Defaults to "inactive" if the user is not found in the graph.

## Machine Learning Model

The script uses a **Random Forest Classifier** to classify users. You can replace this with other algorithms (e.g., SVM, Logistic Regression) by modifying the relevant section in the `main()` function.

### Metrics
The model is evaluated using:
- **Precision**: Accuracy of positive predictions.
- **Recall**: Coverage of actual positive cases.
- **F1-Score**: Harmonic mean of precision and recall.

## Extending the Project

- **Additional Features**: Add graph metrics like PageRank or clustering coefficient.
- **Data Sources**: Integrate with real-world social media data.
- **Improved Models**: Experiment with deep learning methods or graph neural networks.

## Contributing
Feel free to submit issues or pull requests for improvements or feature suggestions.

```text

2025-01-08 23:01:59,951 - INFO - Starting the classification pipeline.
2025-01-08 23:01:59,952 - INFO - Constructing the interaction graph.
2025-01-08 23:01:59,952 - INFO - Graph constructed with 4 nodes and 5 edges.
2025-01-08 23:01:59,952 - INFO - Extracting features from the graph.
2025-01-08 23:01:59,952 - INFO - Features extracted for 4 users.
2025-01-08 23:01:59,953 - INFO - Creating the dataset by merging features with user labels.
2025-01-08 23:01:59,955 - INFO - Dataset created with 4 entries.
2025-01-08 23:01:59,957 - INFO - Training the Random Forest model.
2025-01-08 23:02:00,019 - INFO - Evaluating the model.
2025-01-08 23:02:00,024 - INFO - Model evaluation results:
              precision    recall  f1-score   support

     average       1.00      1.00      1.00         1

    accuracy                           1.00         1
   macro avg       1.00      1.00      1.00         1
weighted avg       1.00      1.00      1.00         1

2025-01-08 23:02:00,024 - INFO - Classifying user: user1
2025-01-08 23:02:00,024 - INFO - Extracting features from the graph.
2025-01-08 23:02:00,024 - INFO - Features extracted for 4 users.
/Users/kuse/Library/Python/3.9/lib/python/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
  warnings.warn(
2025-01-08 23:02:00,029 - INFO - User user1 classified as influential.
2025-01-08 23:02:00,029 - INFO - Final classification for user user1: influential

```