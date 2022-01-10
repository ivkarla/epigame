# pylint: disable=no-self-argument, no-member

from core import GhostSet, GSet, ReprGhost, GRe, meta, struct
from data import np, stats, this, Table, tab
from data import set_repr_to
import pyedflib as edf

_DEBUG = ''

class EEG(Table):
    LABEL_START = 'EEG '
    BAD = ['TTL', 'ECG']
    BP_SEP = '-'
    class time:
        """ converts time units to seconds by frequency sampling (fs) """
        unit = 'units'
        def __init__(_, units): _.time = units
        def __call__(_, fs=None): return _.time/fs
        def __repr__(_): return '{} {}'.format(str(_.time),_.unit)
    class ms(time):
        """ converts ms to time units by frequency sampling (fs) """
        unit = 'ms'
        def __call__(_, fs=1000): return int(round(_.time*fs/1000))
    class secs(time):
        """ converts seconds to time units by frequency sampling (fs) """
        unit = 's'
        def __call__(_, fs=1000): return int(_.time*fs)
    def _load(eeg, epoch, n):
        data = None
        with edf.EdfReader(eeg.file) as file:
            data = [file.readSignal(eeg.labels[id], epoch.at, epoch.span) for id in eeg.labels]
            file.close()
        if data is not None:
            eeg._set(data=np.array(data), at_epoch=(n, epoch()))
        else:
            if 'at_epoch' in eeg.sets: del(eeg.at_epoch)
            eeg._set(data=None)
    class step(GSet):
        START = 0
        CENTRE = 1
        END = 2
        def __init__(step, space, duration):
            step._set(at=space, span=duration)
        def reset(grid, at=0, root=None):
            if root: grid._set(root=root)
            else: root = grid.root
            all_space, left = root.duration(root.fs), 0
            if grid.at.time == 0: epochs = [EEG.step(0, all_space)]
            else:
                space, span, epochs = grid.at(root.fs), grid.span(root.fs), []
                for x in range(at, all_space, space):
                    end = x+span
                    if end>all_space: left = all_space-x
                    else: epochs.append(EEG.step(x, span))
            grid._set(_all=epochs, skip=at, out=left)
        def __call__(step, _as=None):
            if 'root' in meta(step):
                if _as == None: return len(step._all)
            elif _as is not None: step.id = _as
            elif 'id' in meta(step): return step.id
        def items(wrapped):
            if 'root' in meta(wrapped): return wrapped._all
        def __getitem__(by, epoch_n):
            if 'root' in meta(by) and epoch_n<len(by._all):
                by.root._load(by._all[epoch_n], epoch_n)
        def __repr__(_): return '|'.join([repr(_.at),repr(_.span)])
    class event(GSet):
        def __init__(event, to=None, group=None, _as=0, _from=0):
            event._set(mode=_from, note=group, id=_as)
            if to is not None: event.link(to)
        def link(event, to):
            if event.note is None or not 'event' in to.sets:
                if event.note is None:
                    to.event = event
                    event.type = []
                    return
                else: EEG.event(to)
            types = to.event.type
            ids = [to.event.id]+[_type.id for _type in types]
            while event.id in ids: event.id += 1
            if event.note in to.notes:
                event.at = to.notes[event.note]
                types.append(event)
        def __repr__(event):
            _repr = str(event.id)
            if 'at' in meta(event): _repr += ' at: {}'.format(event.at)
            if 'type' in meta(event):
                for subev in event.type: _repr += '; '+repr(subev)
            return _repr
    def _at_last(eeg, sets):
        if 'epoch' in meta(eeg):
            eeg.epoch.reset(root=eeg)
            if len(eeg.axes.time) != eeg.epoch.span(eeg.fs): eeg.axis(eeg, eeg.epoch.span(eeg.fs), 'time', 1)
    @staticmethod
    def from_file(name, step=None, bad=None):
        def correct_(label):
            if label.startswith(EEG.LABEL_START): return label[len(EEG.LABEL_START):]
            return label
        eeg = EEG()
        with edf.EdfReader(name) as file:
            if bad is None: bad = EEG.BAD
            duration = EEG.secs(file.getFileDuration())
            fs = file.getSampleFrequencies()[0]
            if step is None: step = EEG.step(EEG.secs(0), duration)
            raw_notes = file.readAnnotations()
            notes = {note:[] for note in set(raw_notes[-1])}
            for n, note in enumerate(raw_notes[-1]):
                notes[note].append(EEG.secs(raw_notes[0][n]))
            labels = [correct_(label) for label in file.getSignalLabels()]
            labels = {label:n for n,label in enumerate(labels) if label not in bad}
            eeg.set(time=step.span(fs), region=tuple(labels))
            eeg(file=name, duration=duration, fs=fs, notes=notes, labels=labels, epoch=step)
            file.close()
        return eeg
    def remap(eeg, at=None, step=None):
        sets = eeg.sets
        if this(step).inherits(EEG.step): eeg._set(epoch=step)
        if at is None:
            at = eeg._best_map if '_best_map' in sets else 0
        eeg.epoch.reset(at)
        if 'event' in sets:
            deltas = []
            for epoch in eeg.epoch._all: epoch.id = None
            for event in eeg.event.type:
                for time in event.at:
                    at, space, limit = time(eeg.fs), eeg.epoch.at(eeg.fs), len(eeg.epoch.items())-1
                    for n,epoch in enumerate(eeg.epoch.items()):
                        end = epoch.at+space if n<limit else epoch.span
                        if at>=epoch.at and at<end:
                            epoch(event.id)
                            if event.mode == eeg.step.START: deltas.append(EEG.time(at-epoch.at)(eeg.fs))
                            elif event.mode == eeg.step.END: deltas.append(EEG.time(end-at-1)(eeg.fs))
                            else: 
                                centre = epoch.at+int(round(epoch.span/2))
                                deltas.append(EEG.time(abs(at-centre))(eeg.fs))
                            break
            for epoch in eeg.epoch._all:
                if epoch() is None: epoch(eeg.event.id)
            eeg.deltas = deltas
    def optimize(eeg, *events, grid=None):
        if events:
            for event in events: event.link(eeg)
        eeg.remap(0, grid)
        gaussian_space = stats.shapiro if len(eeg.deltas)<=5000 else stats.normaltest
        def test():
            _, p = gaussian_space(eeg.deltas) if len(eeg.deltas)>2 else 0,1
            if p<=0.05: return p, np.median(eeg.deltas)
            return p, np.average(eeg.deltas)    
        (p, best), at, check = test(), 0, eeg.epoch.span(eeg.fs)
        print('optimizing epoch position...', end=' ')
        for _try in range(1, check):
            eeg.remap(_try)
            p, check = test()
            if check<best: p, best, at = p, check, _try
        _test = 'median' if p<0.05 else 'mean'
        print('best frame found at {:.3f}s with a {} delay of {:.3f}s'.format(EEG.time(at)(eeg.fs), _test, EEG.time(best)(eeg.fs)))
        eeg._set(_best_map=at)
    class sampler(GSet, GRe):
        eeg = None
        def __init__(map, root, *reserve, **opts):
            raw, proc = [step() for step in root.epoch.items()], []
            find, key = None, {k:v for k,v in reserve}
            for step in raw:
                if find==step: find=None
                if find is None: proc.append(step)
                else: proc.append(None)
                if step in key: find = key[step]
            key = {k:[] for k in list(set(raw))+[None]}
            for n,id in enumerate(proc): key[id].append(n)
            map._set(eeg=root, key=key, mask=proc, **opts)
        def _at_last(_, sets):
            if 'seed' in sets: np.random.seed(_.seed)
        def set(map, **event_range):
            prev, key = map.key, {}
            for k,deltas in event_range.items():
                if k in prev:
                    seq, key[k] = prev[k], []
                    for item in seq: key[k] += [item+d for d in deltas]
                    for o in prev:
                        if o != k:
                            for e in key[k]:
                                if e in prev[o]: prev[o].pop(prev[o].index(e))
            for k in prev:
                if k not in key: key[k] = prev[k]
            map._set(prev=prev, key=key)
        def get(map, event, times, random_seed=None):
            if random_seed and not 'seed' in meta(map): map._set(seed=random_seed)
            if not 'pool' in meta(map): map._set(pool = {k:map.key[k].copy() for k in map.key})
            resampled, sequence = [], []
            while times:
                if len(map.pool[event])==0: map.pool[event] = map.key[event].copy()
                at = map.pool[event].pop(np.random.randint(len(map.pool[event])))
                map.eeg.epoch[at]
                resampled.append(map.eeg.data)
                sequence.append(at)
                times -= 1
            return resampled, sequence
        def __repr__(_): return _._resize('|'.join([str(id) if id!=None else ' ' for id in _.mask]))
    def tag(event, *a_b, **event_range):
        event._set(sample=event.sampler(event, *a_b))
        event.sample.set(**event_range)
        
STEp = epoch = EEG.step
TIME = EEG.time
SET = EEG.event
secs = EEG.secs
ms = EEG.ms

if _DEBUG == 'load':
    source = "C:\\Users\\omico\\OneDrive\\code\\python\\___src\\PAc\\TN-study\\data\\"
    name = "HDW2016MAR02-PAc.EDF"
    eeg = EEG.from_file(source+name, epoch(ms(250), ms(500)))
elif _DEBUG == 'events':
    source = "C:\\Users\\omico\\OneDrive\\data\\ePAt (epileptogenic area by SEEG patterns)\\raw\\"
    name = "HDW2016MAR02-PAc.EDF"
    ##alternative 1
    #eeg = EEG.from_file(source+name)
    #eeg.optimize(SET(group='EEG inicio'), SET(group='EEG fin', _from=epoch.END), grid=STEp(ms(250), ms(500)))
    #alternative 2, preferred
    eeg = EEG.from_file(source+name, epoch(ms(250), ms(500)))
    SET(eeg, _as='N') #baseline event, which means that everything is not an event is called as 'N'
    SET(eeg, 'EEG inicio', _as = 'S', _from = epoch.START)
    SET(eeg, 'EEG fin', 'E', _from = epoch.END)
    eeg.optimize()
    eeg.remap()
    eeg.tag(('S', 'E'), S=range(-40,0,1), E=range(0,-40,-1))
    N, n = eeg.sample.get('N', 10)
    S, s = eeg.sample.get('S', 10)
    E, e = eeg.sample.get('E', 10)
    print(n)
    print(s)
    print(e)
