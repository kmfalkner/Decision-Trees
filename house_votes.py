import ID3, parse, random
import matplotlib.pyplot as plt

def plot():
    y_axiswithPruning = []
    y_axiswithoutPruning = []
    for i in range(1, 31):
        withPruning = []
        withoutPruning = []
        data = parse.parse("house_votes_84.data")
        for j in range(100):
            random.shuffle(data)
            length = (len(data) - (i * 10)) / 2
            length = (length + (i * 10)) + 0.5
            train = data[:i * 10]
            valid = data[i * 10:int(length)]
            test = data[int(length):]

            tree = ID3.ID3(train, 'democrat')

            ID3.prune(tree, valid)
            acc = ID3.test(tree, train)
            acc = ID3.test(tree, valid)
            acc = ID3.test(tree, test)
            withPruning.append(acc)
            tree = ID3.ID3(train+valid, 'democrat')
            acc = ID3.test(tree, test)
            withoutPruning.append(acc)
            # print(withoutPruning)
            # print(withPruning)
        avewithPruning = sum(withPruning)/len(withPruning)
        avewithoutPruning = sum(withoutPruning)/len(withoutPruning)
        y_axiswithoutPruning.append(avewithoutPruning)
        y_axiswithPruning.append(avewithPruning)
        #print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))
    x_axis = []
    for i in range(30):
        i += 1
        x_axis.append(i * 10)
    plt.plot(x_axis, y_axiswithoutPruning, label = "without pruning")
    plt.plot(x_axis, y_axiswithPruning, label = "with pruning")
    plt.legend()
    plt.show()

