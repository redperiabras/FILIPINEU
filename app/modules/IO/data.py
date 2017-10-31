from nltk.tokenize import word_tokenize
from pathlib import Path

import codecs
import random


class Data:
    total_data = 0

    def __init__(self, loc, mode):
        self.file = codecs.open(loc, mode, "utf-8")
        self.data = [line for line in self.file.readlines()]
        self.file.close()
        self.total_data = len(self.data)

    def tokenized_data(self):
        return [word_tokenize(line) for line in self.data]


def dist_data(data, train_size=70, test_size=20, eval_size=10):

    curr_data_len = total_data_len = len(data)
    train_data = test_data = eval_data = []

    for i in range(1, 4):

        if i is 1:  # training
            n = round(total_data_len * (train_size / 100))
        elif i is 2:  # testing
            n = round(total_data_len * (test_size / 100))
        else:  # evaluation
            n = round(total_data_len * (eval_size / 100))

        for j in range(0, n):
            curr_data_len = len(data)
            pick = random.randint(0, curr_data_len - 1)

            if i is 1:  # training
                train_data.append(data.pop(pick))
            elif i is 2:  # testing
                test_data.append(data.pop(pick))
            else:  # evaluation
                eval_data.append(data.pop(pick))

    return train_data, test_data, eval_data


def export_data(loc, data):
    src_file = codecs.open(loc + "data.src", "w+", "utf-8")
    tgt_file = codecs.open(loc + "data.tgt", "w+", "utf-8")

    for i in range(0, len(data)):
        print("\n")
        print(data[i][0])
        print(data[i][1])

        src_file.write(" ".join(data[i][0]))
        src_file.write("\n")

        tgt_file.write(" ".join(data[i][1]))
        tgt_file.write("\n")

    src_file.close()
    src_file.close()
