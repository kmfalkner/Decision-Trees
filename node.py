class Node:
  def __init__(self):
    # The current most likely classification of the example being analyzed
    # Leaf nodes return this value
    self.label = None 
    
    # The name of the attribute being split on
    # if the node is a leaf, feature is None because there is nothing more to split
    self.feature = None

    self.children = {} # A dictionary of attribute values to child nodes - ex for attribute color : {'brown':Node, 'red':Node}

    self.leaf = False # A boolean for whether the node is at the end of a decision tree branch

    