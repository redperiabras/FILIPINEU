from nltk.tokenize import word_tokenize
from pathlib import Path

import codecs


class File_Handler:

    def __init__(self, loc, mode):
        self.file = codecs.open(loc, mode, "utf-8")
        self.lines = [word_tokenize(line) for line in self.file.readlines()]
        self.total_lines = len(self.lines)
        self.file.close()

    def get_number_of_lines(self):
        return self.total_lines

    def get_lines(self):
        return self.lines


class Data_Export:

    def __init__(self, loc, data):
        # check if data location folders
        self.src_file = codecs.open(loc + "data.src", "w+", "utf-8")
        self.tgt_file = codecs.open(loc + "data.tgt", "w+", "utf-8")

        for i in range(0, len(data)):
            print("\n")
            print(data[i][0])
            print(data[i][1])

            for j in range(0, len(data[i][0])):
                self.src_file.write(
                    data[i][0][j] + " ")
            self.src_file.write("\n")

            for j in range(0, len(data[i][1])):
                self.tgt_file.write(
                    data[i][1][j] + " ")
            self.tgt_file.write("\n")

        self.src_file.close()
        self.src_file.close()
