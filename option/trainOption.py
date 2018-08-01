from option.baseOption import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # important
        self.parser.add_argument('--split_train', type=float, default='0.75', help='% of train') # TODO
        # ignorable
        self.parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')  # TODO
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        self.parser.add_argument('--lr_policy', type=str, default='same', help='learning rate policy: same|lambda|step|plateau')
        self.parser.add_argument('--epoch', type=int, default=11, help='How many epoch?')
        self.parser.add_argument('--propensity', type=str, default='no', help='Use propensity? no|naive|constant')
        self.parser.add_argument('--numerator', type=int, default=1, help='Numerator of propensity weighting')
        # visualization
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='epoch frequency of saving model')