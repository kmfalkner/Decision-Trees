from node import Node
import math
import parse
import copy
import unit_tests

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  # the list of attributes that can be split on
  attribute_list = list(examples[0].keys())
  attribute_list.remove("Class")

  # build the tree
  return ID3_recurse(examples, default, attribute_list)


def ID3_recurse(examples, default, attribute_list):
  '''
  ID3 Helper function. Takes in a subset of data to train on, a default value for missing attributes, 
  and a list of attributes to be split on. 
  '''

  # create a node t for the tree
  node = Node()

  # find the most common value of Target in examples
  target_counts = {}
  for example in examples:
    if example["Class"] not in target_counts:
      target_counts[example["Class"]] = 0
    target_counts[example["Class"]] += 1

  # print(f"classification counts: {str(target_counts)}")
  
  # label for this node is the most common classification of items in the data
  node.label = max(target_counts, key=target_counts.get)

  # if all examples in the dataset are classified the same way, return the node
  if len(target_counts) == 1:
    node.leaf = True
    return node

  # if attributes is empty, return the node - nothing to classify
  if len(target_counts)  == 0:
    node.leaf = True
    return node

  # find the best attribute to split the data... A*
  best_attribute = ""
  best_info_gain = 0
  for attribute in attribute_list:
    info_gain = informationGain(examples, attribute, target_counts)
    
    # if this attribute yields better results, update best choices
    if info_gain > best_info_gain:
      best_info_gain = info_gain
      best_attribute = attribute
  
  # print(f"possible attributes: {str(attribute_list)}")
  # print(f"best attribute to split on: {str(best_attribute)} - This attribute will be removed from child splits")
  # print(f"best info gain: {str(best_info_gain)}")

  # if no best attribute (i.e. all the same or all 0), no point in splitting any further down this branch.
  if best_attribute == "":
    return node
  
  # Copy the list of valid attributes and remove the best attribute from the list for recursive calls
  sub_attribute_list = attribute_list.copy()
  sub_attribute_list.remove(best_attribute)

  # Let A* be the attribute from attributes that best classifies examples in D.
  # Assign t the decision attribute A*.
  node.feature = best_attribute

  # dictionary of possible values of the selected attribute corresponding to items with that value
  attribute_to_examples = {}
  for example in examples:
    # add the example into a category based on the value of the selected attribute
    if example[best_attribute] not in attribute_to_examples: 
      attribute_to_examples[example[best_attribute]] = []
    attribute_to_examples[example[best_attribute]].append(example)

  # For each possible value "a" in A*, do:
  for attribute_value in attribute_to_examples:
    # Add a new tree branch below t, corresponding to the test A* = "a".
    # if the subset of data with the selected attribute value is empty
    # Let D_a be the subset of D that has value "a" for A*
    subset = attribute_to_examples[attribute_value]

    # if there are no examples with this attribute value 
    # or there are no more features left to split on in this branch
    if len(subset) == 0 or len(attribute_list) == 0:
      # don't need to split any further
      sub_node = Node()
      # Label this sub-node with the most common value of Target in D
      sub_node.label = max(target_counts, key=target_counts.get)
      # add this leaf to the parent node
      sub_node.leaf = True
      node.children[attribute_value] = sub_node
    # otherwise need to attach the rest of the tree based on split data
    else:
      sub_node = ID3_recurse(subset, "default", sub_attribute_list)
      node.children[attribute_value] = sub_node
  
  # Return t.
  return node

def entropy(examples, class_counts):
  ''' 
  Takes in an array of examples, and a dictionary with classification types and the 
  number of occurences in the examples with that classification
  '''
  total_entropy = 0
  for count in class_counts.values():
    if count > 0:
      proportion = count / len(examples)
      total_entropy -= proportion * math.log(proportion, 2)
  
  return total_entropy


def informationGain(examples, attribute, class_counts):
  ''' 
  Takes in an array of examples and an attribute name, and a dictionary with classification
  types and the number of occurences in the examples with that classification.
  '''

  # calculate entropy of current data before the split
  parent_entropy = entropy(examples, class_counts)

  # need to classify examples (data) into categories based on the value of the attribute we're looking at
  attribute_to_examples = {}
  for example in examples:
    # add the example into a category based on the value of the selected attribute
    if example[attribute] not in attribute_to_examples: 
      attribute_to_examples[example[attribute]] = []
    attribute_to_examples[example[attribute]].append(example)

  # calculate the entropy of each category of attribute values
  weighted_child_entropy = 0
  for subset in attribute_to_examples.values():
    
    # find the number of occurrences for this category
    subset_class_counts = {}
    for example in subset:
      if example["Class"] not in subset_class_counts:
        subset_class_counts[example["Class"]] = 0
      subset_class_counts[example["Class"]] += 1

    unweighted_entropy = entropy(subset, subset_class_counts)
    #print(f"unweighted entropy: {unweighted_entropy}")
    weighted_child_entropy += (len(subset)/len(examples)) * unweighted_entropy
    #print(f"weighted entropy: {weighted_child_entropy}")

  # return the difference in entropies before and after split -> info gain
  information_gain = parent_entropy - weighted_child_entropy
  return information_gain


def prune(node, validation_data):
    '''
    Takes in a trained tree and a validation set of examples.
    Prunes nodes in order to improve accuracy on the validation data.
    '''
    prune_recurse(node, node, validation_data)

def prune_recurse(root, node, examples):
    # If the node is already a leaf, there's nothing to prune
    if node.leaf:
        return

    # Recursively prune all children first (bottom-up pruning)
    for child_value, child_node in node.children.items():
        prune_recurse(root, child_node, examples)

    # After pruning the children, check if pruning this node improves accuracy
    original_accuracy = test(root, examples)

    # Temporarily prune the node by making it a leaf
    original_children = copy.deepcopy(node.children)  # Deep copy to fully restore later if needed
    node.children = {}  # Prune by setting children to an empty dictionary
    node.leaf = True  # Mark node as a leaf

    # Compute accuracy after pruning
    pruned_accuracy = test(root, examples)

    # If pruning does not improve accuracy, restore the original subtree
    if pruned_accuracy < original_accuracy:
        node.children = original_children  # Restore the original children
        node.leaf = False  # Mark it as an internal node again


def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  correct = 0
  total = 0
  for example in examples:
    eval = evaluate(node, example)
    if eval == example["Class"]:
      correct += 1
    total += 1
  if total == 0:
    return 0
  return correct / total

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  current = node
  current_attribute = current.label

  # descend the decision tree based on the value of the current attribute
  while (not current.leaf):
    # if current is None, return the classification of the most recent node
    if current is None:
      return current_attribute

    # otherwise, the new attribute to assess in our example is the current node's feature
    current_attribute = current.feature

    # If attribute not in example dictionary, then return current.label
    if (current_attribute not in example.keys()):
      return current.label
    
    # the actual value of this attribute
    example_attribute = example[current_attribute]

    # if not a valid value, return whatever the best classification so far is
    if example_attribute not in current.children:
      return current.label
    
    # decide which node to descend to based the value of the example
    current = current.children[example_attribute]
  
  # a leaf in the tree is reached, return the label (classification) at that leaf
  return current.label


def printTree(node, level=0):
  '''
  Prints the tree structure with indentation for better visualization.
  '''
  if node is not None:
    # Print the current node with indentation based on its level in the tree
    indent = "  " * level
    print(f"{indent}{node.feature},{node.label}")

    # Print all the children of the current node recursively
    for child in node.children:
      print(f'{child}:', end='')
      printTree(node.children[child], level + 1)


if __name__ == "__main__":
  examples = parse.parse("cars_train.data")
  tree = ID3(examples, "default")
  printTree(tree)

  unit_tests.testID3AndTest()
  unit_tests.testID3AndEvaluate()
  unit_tests.testPruning()
  # unit_tests.testPruningOnHouseData('house_votes_84.data')