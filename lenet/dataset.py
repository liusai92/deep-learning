"""
generate data to feed the net.
"""

import numpy as np
import csv
import os

class dataloaderbase(object):

    def __init__(self, data_path):
        
        self.file = open(data_path)
        self.reader = csv.reader(self.file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


class dataloader_predict(dataloaderbase):
    """
    generate data from test.csv, where every line is pixel0...pixel783.
    construct writer to submission.csv.
    """
    def __init__(self, data_path):
        super().__init__(data_path)
        self.wfile = open(os.path.dirname(data_path) + '/submission.csv', 'w')
        self.writer = csv.writer(self.wfile)
        self.writer.writerow(['ImageId', 'Label'])

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.wfile.close()

    def parse_one_line(self, line):
        line = list(map(int, line))
        image = np.array(line, dtype=np.float)
        image[image>0] = 1.0
        image = image.reshape([28, 28, 1])
        return image

    def next_line(self):
        image = np.empty([1, 28, 28, 1])
        line = self.reader.__next__()
        if 'pixel' in line[10]:
            line = self.reader.__next__()
        image[0,:,:,:] = self.parse_one_line(line)
        return image

class dataloader_train(dataloaderbase):
    """
    generate data from train.csv, where every line is label, pixel0...pixel783.
    """
    def __init__(self, batch_size, data_path):
        super().__init__(data_path=data_path)
        self.batch_size = batch_size
        self.data_path = data_path

    def parse_one_line(self, line):
        line = list(map(int, line))
        label = int(line[0])
        image = np.array(line[1:], dtype=np.float)
        image[image>0] = 1.0
        image = image.reshape([28, 28, 1])
        return image, label

    def reconnect(self):
        self.file.close()
        self.file = open(self.data_path)
        self.reader = csv.reader(self.file)
       
    def next_batch(self):

        images = np.empty([self.batch_size, 28, 28, 1])
        labels = np.zeros([self.batch_size, 10])

        i = 0
        while i < self.batch_size:
            try:
                line = self.reader.__next__()
                if 'pixel' in line[10]:
                    line = self.reader.__next__()
                image, label = self.parse_one_line(line)
                images[i,:,:,:] = image
                labels[i, label-1] = 1.0
                i += 1
            except StopIteration:
                self.reconnect()
        return images, labels
