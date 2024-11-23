import random

class FeatureSelection:
    def __init__(self, num_features):
        self.num_features = num_features
        
    def random_evaluation(self, features):
        return random.uniform(0, 100)
    
    def forward_selection(self):
        current_features = set()
        best_features = set()
        best_accuracy = self.random_evaluation(current_features)
        
        print(f"Using no features and \"random\" evaluation, I get an accuracy of {best_accuracy:.1f}%")
        print("Beginning search.")
        
        while len(current_features) < self.num_features:
            best_local_accuracy = 0
            best_new_feature = None
            
            for feature in range(1, self.num_features + 1):
                if feature not in current_features:
                    test_features = current_features.copy()
                    test_features.add(feature)
                    
                    accuracy = self.random_evaluation(test_features)
                    print(f"Using feature(s) {sorted(list(test_features))} accuracy is {accuracy:.1f}%")
                    
                    if accuracy > best_local_accuracy:
                        best_local_accuracy = accuracy
                        best_new_feature = feature
            
            if best_new_feature:
                current_features.add(best_new_feature)
                print(f"Feature set {sorted(list(current_features))} was best, accuracy is {best_local_accuracy:.1f}%")
                
                if best_local_accuracy > best_accuracy:
                    best_accuracy = best_local_accuracy
                    best_features = current_features.copy()
                    
        print(f"Finished search!! The best feature subset is {sorted(list(best_features))}, "
              f"which has an accuracy of {best_accuracy:.1f}%")
    
    def backward_elimination(self):
        current_features = set(range(1, self.num_features + 1))
        best_features = current_features.copy()
        best_accuracy = self.random_evaluation(current_features)
        
        print(f"Using all features and \"random\" evaluation, I get an accuracy of {best_accuracy:.1f}%")
        print("Beginning search.")
        
        while len(current_features) > 1:
            best_local_accuracy = 0
            best_feature_to_remove = None
            
            for feature in current_features:
                test_features = current_features.copy()
                test_features.remove(feature)
                
                accuracy = self.random_evaluation(test_features)
                print(f"Using feature(s) {sorted(list(test_features))} accuracy is {accuracy:.1f}%")
                
                if accuracy > best_local_accuracy:
                    best_local_accuracy = accuracy
                    best_feature_to_remove = feature
            
            if best_feature_to_remove:
                current_features.remove(best_feature_to_remove)
                print(f"Feature set {sorted(list(current_features))} was best, accuracy is {best_local_accuracy:.1f}%")
                
                if best_local_accuracy > best_accuracy:
                    best_accuracy = best_local_accuracy
                    best_features = current_features.copy()
                    
        print(f"Finished search!! The best feature subset is {sorted(list(best_features))}, "
              f"which has an accuracy of {best_accuracy:.1f}%")

def main():
    print("Welcome to Feature Selection Algorithm.")
    num_features = int(input("Please enter total number of features: "))
    
    print("\nType the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    
    choice = int(input())
    
    selector = FeatureSelection(num_features)
    
    if choice == 1:
        selector.forward_selection()
    elif choice == 2:
        selector.backward_elimination()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()