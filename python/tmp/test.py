import sys
import random

DEBUG = True


class IO(object):

    def __init__(self):
        if DEBUG:
            self.counter = 0

    def ReadInput(self):
        i = input()
        if DEBUG:
            print('incoming: ' + str(i), file=sys.stderr)
            sys.stderr.flush()
        return i

    def PrintOutput(self, output):
        if DEBUG:
            print(str(self.counter) + ' sending: ' + str(output), file=sys.stderr)
            sys.stderr.flush()
            self.counter += 1
        print(output)
        sys.stdout.flush()

if __name__ == "__main__":
    io = IO()    
    while True:
        #io.PrintOutput(random.choice(['1 1', '1 0', '1 -1', '0 1', '0 0', '0 -1', '-1 1', '-1 0', '-1 -1']))
        io.PrintOutput('1 1')
        io.ReadInput()