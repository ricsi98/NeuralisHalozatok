import sys
import random

DEBUG = True


class IO(object):

    def ReadInput(self):
        i = input()
        if DEBUG:
            print('incoming: ' + str(i), file=sys.stderr)
            sys.stderr.flush()
        return i

    def PrintOutput(self, output, idx=0):
        if DEBUG:
            print(str(idx) + ' sending: ' + str(output), file=sys.stderr)
            sys.stderr.flush()
        print(output)
        sys.stdout.flush()

if __name__ == "__main__":
    io = IO()    
    while True:
        #io.PrintOutput(random.choice(['1 1', '1 0', '1 -1', '0 1', '0 0', '0 -1', '-1 1', '-1 0', '-1 -1']))
        io.PrintOutput('1 0')