
class Analysis():
    def __init__(self):
        super(Analysis).__init__()
        self.prop = []

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
                p = split[2]
                assert p.startswith('p')
                p = p.lstrip('p ').strip()
                propensity = float(p)
                self.prop.append(propensity)
        fin.close()

    def analysis(self):
        clipping = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
        for i in range(20):
            clipping.append((i+1) * 1000)
        tot = len(self.prop)
        percent = []
        print('Average: ' + str(sum(self.prop)/float(tot)))
        print('Max: ' + str(max(self.prop)))
        for k in clipping:
            cnt = len([1 for i in self.prop if i >= k])
            perc = 100.0* (float(cnt)/tot)
            print("[%.2f" % perc + '%] ' + str(cnt) + ' in ' + str(tot) + ' larger than ' + str(k))
            percent.append(perc)
        print(clipping)
        print(percent)

    def name(self):
        return 'General CAI Dataset'

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

analysis = Analysis()
analysis.initialize()
analysis.analysis()
