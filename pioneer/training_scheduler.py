from torch import nn, optim

class TrainingScheduler:
    def __init__(self, opts, session):
        self.configs = []
        self.myphase = 0
        self.opts = opts
        self.m = 0
        self.session = session
        self.phaseChangedOnLastUpdate = False
    def update(self, iteration):
        self.phaseChangedOnLastUpdate = False
        while (self.myphase < len(self.configs)-1 and self.configs[self.myphase+1][0]*1000 <= iteration):
            self.myphase += 1
            self.session.phase = self.configs[self.myphase][1]
            for opt, lr in zip(self.opts, self.configs[self.myphase][2]):
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
            print('Margin updated: {} -> {}'.format(self.m, self.configs[self.myphase][3]))
            self.m = self.configs[self.myphase][3]
            if not self.configs[self.myphase][4] is None:
                self.configs[self.myphase][4]()
            self.phaseChangedOnLastUpdate = True
            print("Session updated phase to {} as iteration is {}".format(self.session.phase, iteration))

    # Give the LR for each optimizer in the same order as when initing this scheduler object
    def add(self, _iteration, _phase, _lr, _margin, _aux_operations):
        self.configs += [(_iteration, _phase, _lr, _margin, _aux_operations)]
        return self
    def get_iteration_of_current_phase(self, iteration):
        first_rule_with_this_session_phase = self.myphase
        while first_rule_with_this_session_phase > 0:
            if self.configs[first_rule_with_this_session_phase-1][1] == self.configs[first_rule_with_this_session_phase][1]:
                first_rule_with_this_session_phase -= 1
            else:
                break
        return iteration - self.configs[first_rule_with_this_session_phase][0]*1000

class TestSession:
    phase = 0
    def __init__(self):
        self.phase=0

def testTrainingScheduler():
    s = TestSession()
    
    nw = nn.Conv2d(3, 512, 1)

    opt1 = optim.Adam(nw.parameters(), 0.0005, betas=(0.0, 0.99))
    opt2 = optim.Adam(nw.parameters(), 0.0001, betas=(0.0, 0.99))
    ts = TrainingScheduler([opt1, opt2], s)

    ts.add(0, 0, 0, 0, None)
    ts.add(9600, _phase=4, _lr=[0.002, 0.0010], _margin=0.05, _aux_operations=None)
    ts.add(14000, _phase=5, _lr=[0.003, 0.0015], _margin=0.06, _aux_operations=None)
    ts.add(20000, _phase=6, _lr=[0.004, 0.0017], _margin=0.07, _aux_operations=None)

    ts.update(9500000)
    assert(ts.m == 0)
    assert(s.phase < 4)
    assert(opt1.param_groups[0]['lr'] == 0.0005)

    ts.update(9600000)
    assert(ts.m == 0.05)
    assert(opt1.param_groups[0]['lr'] == 0.002)
    assert(opt2.param_groups[0]['lr'] == 0.0010)
    assert(s.phase==4)
    ts.update(9600001)
    assert(ts.m == 0.05)
    assert(opt1.param_groups[0]['lr'] == 0.002)
    assert(opt2.param_groups[0]['lr'] == 0.0010)
    assert(s.phase==4)
    ts.update(13999999)
    assert(ts.m == 0.05)
    assert(opt1.param_groups[0]['lr'] == 0.002)
    assert(opt2.param_groups[0]['lr'] == 0.0010)
    assert(s.phase==4)
    ts.update(14000000)
    assert(ts.m == 0.06)
    assert(opt1.param_groups[0]['lr'] == 0.003)
    assert(opt2.param_groups[0]['lr'] == 0.0015)
    assert(s.phase==5)
    ts.update(14000000) #Same, no change
    assert(ts.m == 0.06)
    assert(opt1.param_groups[0]['lr'] == 0.003)
    assert(opt2.param_groups[0]['lr'] == 0.0015)
    assert(s.phase==5)
    ts.update(14000001) #+1 step, no change
    assert(ts.m == 0.06)
    assert(opt1.param_groups[0]['lr'] == 0.003)
    assert(opt2.param_groups[0]['lr'] == 0.0015)
    assert(s.phase==5)
    ts.update(20000001) #next phase
    assert(ts.m == 0.07)
    assert(opt1.param_groups[0]['lr'] == 0.004)
    assert(opt2.param_groups[0]['lr'] == 0.0017)
    assert(s.phase==6)


#testTrainingScheduler()
