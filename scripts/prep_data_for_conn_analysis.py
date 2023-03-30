"""

This notebook loads EDF files and prepares struct objects, containing:

x -raw epochs,

i - epoch indices,

y - epoch labels (1-seizure epoch; 0-basleine epoch),

x_prep - prepreprocessed epochs,

nodes - nodes labels,

extra_nodes_1 - list of extra channels in seizure recording,

extra_nodes_0 - list of extra channels in baseline recording.

The objects are saved with filename format N_woi_bands.prep (N - subject ID; woi - window of interest; bands - filtered frequency)

"""



from src.eeg import EEG, SET, epoch, ms, np, struct
from src.data_legacy import band
from src.connectivity import preprocess

from os import makedirs
from glob import glob

from inspect import isfunction, ismethod, isgeneratorfunction, isgenerator, isroutine
from inspect import isabstract, isclass, ismodule, istraceback, isframe, iscode, isbuiltin
from inspect import ismethoddescriptor, isdatadescriptor, isgetsetdescriptor, ismemberdescriptor
from inspect import isawaitable, iscoroutinefunction, iscoroutine

from collections.abc import Iterable as iterable

from pickle import load, dump

def isfx(field): return ismethod(field) or isfunction(field)

class GhostSet:
    """ enhanced interface (ghost) to retrieve class fields """
    def _meta(data): return {k:v for k,v in data.__dict__.items() if not isfx(v)}
    def _at_last(_, sets): pass
    def _set(object, **sets):
        ''' use to fast initialize fields | needed to avoid initialization problems at copy by value '''
        for field in sets: setattr(object, field, sets[field])
        object._at_last(sets)
GSet = GhostSet

def meta(object):
    ''' retrieves clonable object metadata (__dict__) as a copy '''
    if isinstance(object, GSet): return object._meta()
    return {}

class ClonableObjectGhost:
    """ enhanced interface (ghost) for clonable objects """
    def _by_val(_, depth=-1, _layer=0): pass
GCo = ClonableObjectGhost

class ClonableObject(GSet, GCo):
    """ base clonable object """
    def __init__(this, **data): this._set(**data)
    def __call__(_, **options): _._set(**options)
    def _by_val(_, depth=-1, _layer=0):
        copy = type(_)()
        copy._set(**_._meta())
        if depth<0 or depth>_layer:
            for field in copy.__dict__:
                if isinstance(copy.__dict__[field], ClonableObjectGhost):
                    copy.__dict__[field] = copy.__dict__[field]._by_val(depth,_layer+1)
        return copy
COb = ClonableObject

def copy_by_val(object, depth=-1, _layer=0):
    if isinstance(object, GCo): return object._by_val(depth,_layer)
    return object
copy = by_val = vof = copy_by_val

class ComparableGhost:
    """ enhanced interface (ghost) for comparing instances """
    def _compare(a, b):
        if type(a) != type(b): return False
        if a.__dict__ == b.__dict__: return True
        return False
    def __eq__(a, b): return a._compare(b)
GEq = ComparableGhost

class IterableObjectGhost(GSet):
    """ enhanced interface (ghost) for iterables: exposes __dict__,
        therefore Iterable Objects are like lua dictionaries """
    def __contains__(this, key): return key in this.__dict__
    def __iter__(this): return iter(this.__dict__)
    def items(my): return my.__dict__.items()
    def __getitem__(by, field): return by.__dict__[field]
    def __setitem__(by, field, value): by.__dict__[field] = value
    def pop(by, field): return by.__dict__.pop(field)
GIo = IterableObjectGhost

class ReprGhost:
    """ enhanced interface (ghost) for the skeleton method _repr,
        see implementation of Struct for a working example;
        Record __repr__ override uses _lines_ for max lines display """
    _lines_ = 31
    _chars_ = 13
    _msgsz_ = 62
    _ellipsis_ = ' ... '
    def _repr(my, value):
        _type = ''.join(''.join(str(type(value)).split('class ')).split("'"))
        _value = '{}'.format(value)
        if len(_value)>my._chars_:
            show = int(my._chars_/2)
            _value = _value[:show]+my._ellipsis_+_value[-show:]
        return '{} {}'.format(_type, _value)
    def _resize(this, message, at=.7):
        if len(message)>this._msgsz_:
            start = int(at*this._msgsz_)
            end = this._msgsz_-start
            return message[:start]+this._ellipsis_+message[-end:]
        return message
GRe = ReprGhost

def set_repr_to(lines): GRe._lines_ = lines

class Struct(COb, GEq, GIo, GRe):
    """ structured autoprintable object, behaves like a lua dictionary """
    def __repr__(_):
        return '\n'.join(['{}:\t{}'.format(k, _._repr(v)) for k,v in _.items()])
struct = Struct

class RecordableGhost:
    """ enhanced interface (ghost) for type recording,
        see Record for a working example """
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file: return load(file)
    def save(data, filename):
        with open(filename, 'wb') as file: dump(data, file)
        
GRec = RecordableGhost

class Record(GSet, GCo, GRec, GEq, GRe):
    """ wrapper for any object or value, auto-inspects and provides load/save type structure """
    data = None
    _check = dict(
            isfunction=isfunction, ismethod=ismethod, isgeneratorfunction=isgeneratorfunction, isgenerator=isgenerator, isroutine=isroutine,
            isabstract=isabstract, isclass=isclass, ismodule=ismodule, istraceback=istraceback, isframe=isframe, iscode=iscode, isbuiltin=isbuiltin,
            ismethoddescriptor=ismethoddescriptor, isdatadescriptor=isdatadescriptor, isgetsetdescriptor=isgetsetdescriptor, ismemberdescriptor=ismemberdescriptor,
            isawaitable=isawaitable, iscoroutinefunction=iscoroutinefunction, iscoroutine=iscoroutine
                   )
    def __init__(this, token, **meta):
        this.data = token
        this.__dict__.update({k:v(token) for k,v in this._check.items()})
        super()._set(**meta)
    @property
    def type(_): return type(_.data)
    def inherits(_, *types): return issubclass(_.type, types)
    @property
    def isbaseiterable(_): return _.inherits(tuple, list, dict, set) or _.isgenerator or _.isgeneratorfunction
    @property
    def isiterable(_): return isinstance(_.data, iterable) and _.type is not str
    def _clone_iterable(_):
        if _.inherits(dict): return _.data.copy()
        elif _.isgenerator or _.isgeneratorfunction: return (i for i in list(_.data))
        else: return type(_.data)(list(_.data)[:])
    def _meta(data): return {k:v for k,v in data.__dict__.items() if k != 'data' and not isfx(v)}
    def _by_val(_, depth=-1, layer=0):
        data = _.data
        if _.isiterable: data = _._clone_iterable()
        elif _.inherits(ClonableObjectGhost): data = by_val(data, depth, layer)
        return type(_)(data, **meta(_))
    def __enter__(self): self._instance = self; return self
    def __exit__(self, type, value, traceback): self._instance = None
    def __repr__(self):
        if not hasattr(self, '_preprint'): return Record(self.data, _preprint='', _lines=Record(Record._lines_)).__repr__()
        if self.isbaseiterable:
            pre, repr = self._preprint, ''
            for n,i in enumerate(self.data):
                if self._lines.data == 0: break
                else: self._lines.data -= 1
                index, item = str(n), i
                if self.inherits(dict): index += ' ({})'.format(str(i)); item = self.data[i]
                repr += pre+'{}: '.format(index)
                next = Record(item, _preprint=pre+'\t', _lines=self._lines)
                if next.isiterable: repr += '\n'
                repr += next.__repr__()
                repr += '\n'
            return repr
        elif self.inherits(GCo): return Record(self.data._meta(), _preprint=self._preprint, _lines=self._lines).__repr__()
        else: return self._repr(self.data)
REc = Record

class Bisect(list, COb):
    """ bisect implementation using clonable objects """
    def __init__(set, *items, key=None, reverse=False):
        if not key: key = lambda  x:x
        super().__init__(sorted(items, reverse=reverse, key=key))
    def _bisect(set, item, key, reverse, bottom, top):
        def _(check):
            if key: return key(check)
            return check
        at = int((top-bottom)/2)+bottom
        if len(set)==0: return (0,-1)
        if item==_(set[at]): return (at,0)
        bigger = item<_(set[at])
        if bigger != reverse:
            if at-bottom>0: return set._bisect(item, key, reverse, bottom, at)
            return (at,-1)
        elif top-at>1: return set._bisect(item, key, reverse, at, top)
        return (at,1)
    def search(_, item, key=None, reverse=False):
        if not key: key = lambda x:x
        return _._bisect(item, key, reverse, 0, len(_))
    def _by_val(_, depth=-1, _layer=0):
        copy = super()._by_val(depth, _layer)
        copy += _[:]
        return copy
BSx = Bisect

# ----------------------------------------------------------------------------------------------------------------------------------

main_folder = "/home/kivi/gdrive/epigame-folder/"

woi_code = {'2':"preseizure5", '3':"preseizure4", '4':"preseizure3", '5':"preseizure2", '6':"preseizure1", '7':"transition1", '8':"transition2", '9':"transition60"}
wois = list(woi_code.keys())

# subject_fs = {'VBM':250, 'JQN':500, 'BGL':500, 'HDW':500, 'ASJ':500, 'SDA':500, 'MGM':500, 'MSF':500, 'PTD':500, 'RGE':500, 'SRM':500, 'CRD':500, 'CRF':500, 'GTA':500, 'HAF':500, 'VML':1024, 'MRI':500, 'USA':512, 'BRM':512, 'MMM':2048, 'VCG':500}
subject_fs = {'MMM':2048}
subject_acronyms = list(subject_fs.keys())
subject_ids = {
"ASJ":1,
"BGL":2,
"BRM":3,
"CRD":4,
"CRF":5,
"GTA":6,
"HAF":7,
"HDW":8,
"JQN":9,
"MGM":10,
"MMM":11,
"MRI":12,
"MSF":13,
"PTD":14,
"RGE":15,
"SDA":16,
"SRM":17,
"USA":18,
"VBM":19,
"VCG":20,
"VML":21
}

for woi in wois:
    print(woi_code[woi])

    for bands in [None,(0,4),(4,8),(8,13),(13,30),(30,70),(70,150)]:
        print(bands)

        for sub in subject_acronyms:
            
            file_seizure = glob(main_folder + f"data/{sub}*-seizure.EDF")[0]
            file_baseline =  glob(main_folder + f"data/{sub}*-baseline.EDF")[0]

            print("Subject ID:", sub)

            span, step = 1000, 500      # in ms
            min_woi_duration = 60000    # in ms
            n_epochs = int((min_woi_duration/step)-1)

            print("Number of epochs to consider for classification =", n_epochs)

            eeg_seizure = EEG.from_file(file_seizure, epoch(ms(step), ms(span)))    # load raw seizure SEEG data as an EEG object (class) 
            eeg_baseline = EEG.from_file(file_baseline, epoch(ms(step), ms(span)))   # load raw baseline SEEG data as an EEG object (class)    

            notes_seizure = [note for note in eeg_seizure.notes]
            notes_baseline = [note for note in eeg_baseline.notes]

            sz_start_note, sz_end_note, base_center_note, base_end_note = 'EEG inicio', 'EEG fin', 'mitad-NS', 'NS-fin'

            if sz_start_note not in notes_seizure:
                altnote = [a for a in notes_seizure if sz_start_note in a]
                print(f"{sz_start_note} not in seizure recording notes; alternative note found: {altnote}")
                sz_start_note = altnote[0]
            if sz_end_note not in notes_seizure:
                altnote = [a for a in notes_seizure if sz_end_note in a]
                print(f"{sz_end_note} not in seizure recording notes; alternative note found: {altnote}")
                sz_end_note = altnote[0]
            if base_center_note not in notes_baseline:
                altnote = [a for a in notes_baseline if base_center_note in a]
                print(f"{base_center_note} not in seizure recording notes; alternative note found: {altnote}")
            if base_end_note not in notes_baseline:
                altnote = [a for a in notes_baseline if base_end_note in a]
                print(f"{base_end_note} not in seizure recording notes; alternative note found: {altnote}")

            nodes_seizure = list(eeg_seizure.axes.region)
            nodes_baseline = list(eeg_baseline.axes.region)

            montage_overlap = list(set(nodes_seizure) & set(nodes_baseline))

            extra_in_baseline = [ch for ch in nodes_baseline if ch not in montage_overlap]
            extra_in_seizure = [ch for ch in nodes_seizure if ch not in montage_overlap]

            if not extra_in_baseline and not extra_in_seizure: print(f"\nEEG channels (nodes) match between the seizure and baseline recordings ({len(montage_overlap)} nodes).")

            if extra_in_baseline: 
                print(f"\nExtra nodes in baseline recording ({len(nodes_baseline)} total): {extra_in_baseline}")
                for chn in extra_in_baseline: eeg_baseline.axes.region.remove(chn)

            if extra_in_seizure: 
                print(f"\nExtra nodes in seizure recording ({len(nodes_seizure)} total): {extra_in_seizure}")
                for chn in extra_in_seizure: eeg_seizure.axes.region.remove(chn)

            print(f"Number of common nodes = {len(montage_overlap)}")

            nodes = montage_overlap

            eeg_seizure._set(fs = subject_fs[sub])
            eeg_baseline._set(fs = subject_fs[sub])

            fs_min = min(eeg_seizure.fs, eeg_baseline.fs)
            resampling = 512 if fs_min==512 else 500

            SET(eeg_seizure, _as='N')                      # N - baseline (non-seizure)
            SET(eeg_seizure, sz_start_note, 'W')            # W - WOI
            SET(eeg_seizure, sz_end_note, 'S', epoch.END)    # S - seizure

            SET(eeg_baseline, _as='N')
            SET(eeg_baseline, base_center_note, 'W')            # W - middle point
            SET(eeg_baseline, base_end_note, 'S', epoch.END)    # S - terminal point (end of recording)

            eeg_seizure.optimize()
            eeg_seizure.remap()

            eeg_baseline.optimize()
            eeg_baseline.remap()

            units = int((eeg_seizure.notes[sz_start_note][0].time - eeg_seizure.notes[sz_end_note][0].time)*(span/step))

            if woi == "1":
                woi_start = -units
                woi_end = 0

            elif woi in [str(n) for n in [2,3,4,5,6]]:
                woi_start = - int(woi_code[woi][-1])*n_epochs
                woi_end = - (int(woi_code[woi][-1])-1)*n_epochs

            elif woi in [str(n) for n in [7,8]]:
                woi_start = - int(round(int(woi_code[woi][-1])*60/2))
                woi_end = - woi_start

            elif woi == "9":
                woi_start = - int(round(units*.3))
                woi_end = - woi_start

            elif woi == "10":
                woi_start = -1
                woi_end = 0

            eeg_seizure.tag(('W', 'S'), W=range(int(woi_start),int(woi_end),1), S=range(0,-units,-1))

            eeg_baseline.tag(('W', 'S'), W=range(int(woi_start),int(woi_end),1), S=range(0,-units,-1))

            a, ai = eeg_seizure.sample.get('W', n_epochs)
            b, bi = eeg_baseline.sample.get('W', n_epochs)
            i = ai + bi
            x = a + b
            y = [1]*n_epochs + [0]*n_epochs
            print("Total number of epochs (seizure + baseline) =", len(x))

            pp_seizure = [preprocess(eeg_seizure, ep, resampling) for i,ep in enumerate(a)] 
            print("Resampled to", pp_seizure[0].shape)

            pp_baseline = [preprocess(eeg_baseline, ep, resampling) for i,ep in enumerate(b)] 
            print("Resampled to", pp_baseline[0].shape)

            fpp_seizure = [band(e, bands, pp_seizure[0].shape[1]) for e in pp_seizure] if bands is not None else pp_seizure
            fpp_baseline = [band(e, bands, pp_baseline[0].shape[1]) for e in pp_baseline] if bands is not None else pp_baseline

            ready_epochs = fpp_seizure + fpp_baseline

            prep = struct(x=np.array(x), y=np.array(y), i=np.array(i), x_prep=ready_epochs, nodes=nodes, extra_nodes_1=extra_in_seizure, extra_nodes_0=extra_in_baseline)

            path_prep = main_folder + "preprocessed/"
            makedirs(path_prep, exist_ok=True)

            if bands is not None:          REc(prep).save(path_prep + f"{subject_ids[sub]}-{woi_code[woi]}-{bands}.prep".replace(" ",""))
            elif bands is None:    REc(prep).save(path_prep + f"{subject_ids[sub]}-{woi_code[woi]}.prep")
