import ID3, parse

train_data = ["training", parse.parse("cars_train.data")]
test_data = ["testing", parse.parse("cars_test.data")]
valid_data = ["validation", parse.parse("cars_valid.data")]

tree = ID3.ID3(train_data[1], 0)

for data in [train_data, test_data, valid_data]:
    acc = ID3.test(tree, data[1])
    print(f"{data[0]} accuracy before pruning: {acc}")

ID3.prune(tree, valid_data[1])

for data in [train_data, test_data, valid_data]:
    acc = ID3.test(tree, data[1])
    print(f"{data[0]} accuracy after pruning: {acc}")
