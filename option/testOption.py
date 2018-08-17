from option.baseOption import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')  # TODO
        self.parser.add_argument('--nopost', action='store_true', help='Do not do post processing')
        self.isTrain = False
