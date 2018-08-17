
class GetAvg():
    def __init__(self):
        super(GetAvg).__init__()
        self.res = []

    def initialize(self):
        fin = open('../datasets/criteo-CAI-full/test.txt', 'r')
        print('Initializing Dataset...')
        cnt = 0
        nowid = None
        nowidcnt = 0
        for line in fin:
            if line == '':
                break
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)
            split = line.split('|')
            id = int(split[0].strip())

            if nowid is None:
                nowid = id
            if nowid != id:
                self.res.append({'id': nowid, 'cnt': nowidcnt})
                nowid = id
                nowidcnt = 0
            nowidcnt += 1
        self.res.append({'id': nowid, 'cnt': nowidcnt})
        fin.close()

    def generate(self):
        with open('pred.txt', 'w+') as output:
            for res in self.res:
                st = ''
                for k in range(res['cnt']):
                    st += str(k) + ':200000.00'
                    if k != res['cnt'] - 1:
                        st += ','
                output.write("%d;%s\n"%(res['id'], st))

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

get = GetAvg()
get.initialize()
get.generate()
