from itertools import groupby

class Analysis():
    def __init__(self):
        super(Analysis).__init__()
        self.feature = []
        self.dict = {}

    def initialize(self):
        fin = open('../datasets/criteo-CAI-full/train.txt', 'r')
        print('Initializing Dataset...')
        cnt = 0
        for line in fin:
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)
            split = line.split('|')
            id = int(split[0].strip())
            if len(split) == 4:
                f = split[3].lstrip('f ').strip()
                f0, f1, idx, val = self.parse_features(f)
                self.feature.append({'id': id, 'f': str(idx)})
                if not str(idx) in self.dict:
                    self.dict[str(idx)] = 1
                else:
                    self.dict[str(idx)] += 1
        fin.close()

    def analysis(self):
        lengths = []
        with open('validation.txt', 'w+') as output:
            groups = groupby(self.feature, key=lambda x: x['f'])
            for id, group in groups:
                arrayls = []
                for item in group:
                    arrayls.append(item)
                if len(arrayls) > 1:
                    for k in arrayls:
                        output.write(str(k) + '\n')
                    lengths.append(len(arrayls))
        with open('validation.txt', 'a+') as output:
            print(lengths)
            output.write(str(lengths) + '\n')
            print('Sum: ' + str(sum(lengths)))
            output.write('Sum: ' + str(sum(lengths)) + '\n')
            print('groups: ' + str(len(lengths)))
            output.write('groups: ' + str(len(lengths)) + '\n')

        lengths = []
        with open('validation2.txt', 'w+') as output2:
            for key in self.dict.keys():
                if self.dict[key] > 1:
                    lengths.append(self.dict[key])
            print(lengths)
            print(str(sum(lengths)))
            print(str(max(lengths)))
            output2.write(str(lengths))


    def parse_features(self, s):
        split = s.split(' ')
        f0 = split[0]
        assert f0.startswith('0:')
        f0 = int(f0[2:])

        f1 = split[1]
        assert f1.startswith('1:')
        f1 = int(f1[2:])

        idx = []
        values = []

        for fv in split[2:]:
            f, v = fv.split(':')
            idx.append(int(f) - 2)
            values.append(int(v))

        return f0, f1, idx, values

    def name(self):
        return 'General CAI Dataset'

analysis = Analysis()
analysis.initialize()
analysis.analysis()
