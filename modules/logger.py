from datetime import datetime

import os
import sys
import logging as log

class Logger:

    def __init__(self):
        with open('logo.txt', 'r') as f:
            text = f.read()
            for line in text.split('\n'):
                print(line)
                f.close()

        log.basicConfig(filename='system.log', level=log.DEBUG)
        self.base = 'FILIPINEU [%s]: '

    def __call__(self, message=None):
        if message is not None:
            return self.base % str(datetime.now()) + message
        return self.base % str(datetime.now())

    def information(self, message):
        for i in range(0, len(message), 40):
            text = self.base % str(datetime.now()) + message[0+i:40+i]
            log.info(text)
            print(text, file=sys.stderr, flush=True)


    def warning(self, message):
        for i in range(0, len(message), 40):
            text = self.base % str(datetime.now()) + message[0+i:40+i]
            log.warning(text)
            print(text, file=sys.stderr, flush=True)