import random
import numpy as np
import time

class Classifier:
    def __init__(self):
        self.training_data = []
        self.training_labels = []
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def train(self, data, labels):
        self.training_data = data
        self.training_labels = labels

    def test(self, test_point):
        min_distance = float('inf')
        nearest_label = None

        for i, train_point in enumerate(self.training_data):
            distance = self.euclidean_distance(test_point, train_point)

            if distance < min_distance:
                min_distance = distance
                nearest_label = self.training_labels[i]

        return nearest_label

class Validator:
    def __init__(self):
        self.classifier = Classifier()

    def validate(self, data, labels, feature_subset=None):
        if feature_subset is not None:
            data = data[:, feature_subset]

        num_instances = len(labels)
        correct_predictions = 0

        for i in range(num_instances):
            train_data = np.delete(data, i, axis=0)
            train_labels = np.delete(labels, i)

            test_point = data[i]
            true_label = labels[i]

            self.classifier.train(train_data, train_labels)
            predicted_label = self.classifier.test(test_point)

            if predicted_label == true_label:
                correct_predictions += 1

        return correct_predictions / num_instances

class FeatureSelection:
    def __init__(self, num_features, data, labels):
        self.num_features = num_features
        self.data = data
        self.labels = labels
        self.validator = Validator()

    def forward_selection(self):
        current_features = set()
        best_features = set()
        best_accuracy = 0

        print(f"Using no features and leave-one-out validation, I get an accuracy of {best_accuracy:.4f}")
        print("Beginning search.")

        while len(current_features) < self.num_features:
            best_local_accuracy = 0
            best_new_feature = None

            for feature in range(1, self.num_features + 1):
                if feature not in current_features:
                    test_features = current_features | {feature}
                    accuracy = self.validator.validate(self.data, self.labels, feature_subset=[f - 1 for f in test_features])

                    print(f"Using feature(s) {sorted(list(test_features))} accuracy is {accuracy:.4f}")

                    if accuracy > best_local_accuracy:
                        best_local_accuracy = accuracy
                        best_new_feature = feature

            if best_new_feature is not None:
                current_features.add(best_new_feature)
                print(f"Feature set {sorted(list(current_features))} was best, accuracy is {best_local_accuracy:.4f}")

                if best_local_accuracy > best_accuracy:
                    best_accuracy = best_local_accuracy
                    best_features = current_features.copy()

        print(f"Finished search!! The best feature subset is {sorted(list(best_features))}, which has an accuracy of {best_accuracy:.4f}")

    def backward_elimination(self):
        current_features = set(range(1, self.num_features + 1))
        best_features = current_features.copy()
        best_accuracy = self.validator.validate(self.data, self.labels, feature_subset=[f - 1 for f in current_features])

        print(f"Using all features and leave-one-out validation, I get an accuracy of {best_accuracy:.4f}")
        print("Beginning search.")

        while len(current_features) > 1:
            best_local_accuracy = 0
            best_feature_to_remove = None

            for feature in current_features:
                test_features = current_features - {feature}
                accuracy = self.validator.validate(self.data, self.labels, feature_subset=[f - 1 for f in test_features])

                print(f"Using feature(s) {sorted(list(test_features))} accuracy is {accuracy:.4f}")

                if accuracy > best_local_accuracy:
                    best_local_accuracy = accuracy
                    best_feature_to_remove = feature

            if best_feature_to_remove is not None:
                current_features.remove(best_feature_to_remove)
                print(f"Feature set {sorted(list(current_features))} was best, accuracy is {best_local_accuracy:.4f}")

                if best_local_accuracy > best_accuracy:
                    best_accuracy = best_local_accuracy
                    best_features = current_features.copy()

        print(f"Finished search!! The best feature subset is {sorted(list(best_features))}, which has an accuracy of {best_accuracy:.4f}")

def load_data(file):
    data = np.loadtxt(file)

    labels = data[:, 0].astype(int)
    features = data[:, 1:]

    normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)

    return normalized_features, labels

def main():
    print("Feature Selection Algorithm")
    dataset_file = input("Enter the dataset file name: ")
    algorithm_choice = input("Choose an algorithm:\n1. Forward Selection\n2. Backward Elimination\nEnter 1 or 2: ")

    data, labels = load_data(dataset_file)
    num_features = data.shape[1]

    feature_selector = FeatureSelection(num_features, data, labels)

    if algorithm_choice == "1":
        print("Running Forward Selection...")
        feature_selector.forward_selection()
    elif algorithm_choice == "2":
        print("Running Backward Elimination...")
        feature_selector.backward_elimination()
    else:
        print("Invalid choice. Please run the program again and select 1 or 2.")



if __name__ == "__main__":
    main()