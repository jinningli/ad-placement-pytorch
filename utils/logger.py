
class Logger(object):
    def __init__(self,  initstd, file):
        self.terminal = initstd
        self.log = file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass