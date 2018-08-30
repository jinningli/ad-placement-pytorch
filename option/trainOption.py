from option.baseOption import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')  # TODO
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        self.parser.add_argument('--lr_policy', type=str, default='same', help='learning rate policy: same|lambda|step|plateau')
        self.parser.add_argument('--epoch', type=int, default=11, help='How many epoch?')
        self.parser.add_argument('--clip_value', type=float, default=1.0, help='Value used for propensity weighting')
        self.parser.add_argument('--piw_gradient', action='store_true', help='Put the piw inside the network?')
        # visualization
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='epoch frequency of saving model')