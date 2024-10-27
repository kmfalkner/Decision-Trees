import parse
import random
import ID3

candy_data = parse.parse("candy.data")

#Method that trains and returns a random forest
def get_random_forest(examples, num_trees, sample_size):
    random_forest = []
    for i in range(num_trees):
        random_subset = random.choices(examples, k=sample_size)
        random_forest.append(ID3.ID3(random_subset, 0))
    return random_forest

#Function for getting a prediction from a given random forest
def evaluate_forest(random_forest, example):
    outcomes = {}
    for tree in random_forest:
        curr_outcome = ID3.evaluate(tree, example)
        if curr_outcome not in outcomes:
            outcomes[curr_outcome] = 0
        outcomes[curr_outcome] += 1

    most_frequent_outcome_count = 0
    most_frequent_outcome = None
    for outcome in outcomes:
        if outcomes[outcome] > most_frequent_outcome_count:
            most_frequent_outcome_count = outcomes[outcome]
            most_frequent_outcome = outcome
    
    return most_frequent_outcome



#Splitting data into train, validation, and test sets
random.shuffle(candy_data)
train_candy = candy_data[:len(candy_data)//2]
valid_candy = candy_data[len(candy_data)//2:3*len(candy_data)//4]
test_candy = candy_data[3*len(candy_data)//4:]

#Training random forest on training set
candy_rf = get_random_forest(train_candy, 20, 30)
#Training single tree on training set
candy_decision_tree = ID3.ID3(train_candy, None)
#ID3.prune(candy_decision_tree, valid_candy)

#Assessing performance on training set
train_correct_predictions_rf = 0
train_correct_predictions_single_tree = 0
for example in train_candy:
    rf_prediction = evaluate_forest(candy_rf, example)
    single_tree_prediction = ID3.evaluate(candy_decision_tree, example)
    if rf_prediction == example["Class"]:
        train_correct_predictions_rf += 1
    if single_tree_prediction == example["Class"]:
        train_correct_predictions_single_tree += 1

#Testing performance on validation set
valid_correct_predictions_rf = 0
valid_correct_predictions_single_tree = 0
for example in valid_candy:
    rf_prediction = evaluate_forest(candy_rf, example)
    single_tree_prediction = ID3.evaluate(candy_decision_tree, example)
    if rf_prediction == example["Class"]:
        valid_correct_predictions_rf += 1
    if single_tree_prediction == example["Class"]:
        valid_correct_predictions_single_tree += 1

#Testing performance on test set
test_correct_predictions_rf = 0
test_correct_predictions_single_tree = 0
for example in test_candy:
    rf_prediction = evaluate_forest(candy_rf, example)
    single_tree_prediction = ID3.evaluate(candy_decision_tree, example)
    if rf_prediction == example["Class"]:
        test_correct_predictions_rf += 1
    if single_tree_prediction == example["Class"]:
        test_correct_predictions_single_tree += 1
    
#Printing accuracy results
print("Single Tree Accuracy (train set): " + str(train_correct_predictions_single_tree / len(train_candy)))
print("Single Tree Accuracy (validation set): " + str(valid_correct_predictions_single_tree / len(valid_candy)))
print("Single Tree Accuracy (test set): " + str(test_correct_predictions_single_tree / len(test_candy)))
print("-------------------------------------------")
print("Random Forest Accuracy (train set): " + str(train_correct_predictions_rf / len(train_candy)))
print("Random Forest Accuracy (validation set): " + str(valid_correct_predictions_rf / len(valid_candy)))
print("Random Forest Accuracy (test set): " + str(test_correct_predictions_rf / len(test_candy)))

