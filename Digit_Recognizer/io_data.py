import csv
import numpy as np

def get_train_data(one_hot = True):
    train_x = []
    labels = []

    with open('data/train.csv', 'r') as f:
        next(f, None)
        reader = csv.reader(f)
        for row in reader:
            train_x.append(row[1:])
            labels.append(row[0])

    #train_x = [int(x) for x in row for row in train_x]
    #labels = [int(x) for x in labels]
    train_x = np.array(train_x, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    train_x = np.multiply(train_x, 1.0 / 255.0)
    if one_hot:
        a = np.array(labels)
        b = np.zeros((len(labels), 10))
        b[np.arange(len(labels)), a] = 1
        train_y = b
    else:
        train_y = labels

    return train_x, train_y

def get_test_data():
    test_x = []

    with open('data/test.csv', 'r') as f:
        next(f, None)
        reader = csv.reader(f)
        for row in reader:
            test_x.append(row)

    return test_x

def save_result(pred_y):
    with open('data/result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "Label"])
        for idx, y in zip(range(1, len(pred_y)+1), pred_y):
            writer.writerow([idx, y])