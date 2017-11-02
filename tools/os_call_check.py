import os

class OsC:
    # tiny checker to use at debugging, will print
    # instead of executing the commands
    def __init__(self, check=True):
        self.check = check

    def call(self, s):
        if self.check:
            print s
        else:
            os.system(s)
