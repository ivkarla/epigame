from inspect import isfunction, ismethod, isgeneratorfunction, isgenerator, isroutine
from inspect import isabstract, isclass, ismodule, istraceback, isframe, iscode, isbuiltin
from inspect import ismethoddescriptor, isdatadescriptor, isgetsetdescriptor, ismemberdescriptor
from datetime import timedelta as _time
from datetime import datetime
from collections.abc import Iterable as iterable

def some(field): return (field is not None and field != [] and field != {} and field != ()) or field == True
def no(field): return not some(field) or field==False or field==''

class class_of:
    _instance = None
    def __init__(_, object):
        _._is = type(object)
    def inherits(_, *types):
        return issubclass(_._is, types)
    def has(_, *types): return _.inherits(*types)
    def __enter__(self):
        self._instance = self
        return self
    def __exit__(self, type, value, traceback): self._instance = None
    @staticmethod
    def each_in(list):
        if isiterable(list):
            return [type(item) for item in list]

class struct:
    def __init__(table, **sets): table.__dict__.update(sets)
    @property
    def sets(this): return set(dir(this)) - set(dir(type(this)))
    def set(object, **fields):
        for field in fields: setattr(object, field, fields[field])
    def get(object, *fields): return [getattr(object, field) for field in fields if field in object.__dict__]
    def _clonable(set, mask=None):
        check = set.__dict__.copy()
        clonable = check.copy()
        if some(mask): pass
#            for field in check:
#                if sum([int(_(check[field])) for _ in mask])+sum([int(_(field)) for _ in mask]): clonable.pop(field)
        return clonable
    @staticmethod
    def _from(type):
        if hasattr(type, '__dict__'): return struct(**type.__dict__.copy())
        return struct()

def meta(data, *mask): return struct._from(data)._clonable(mask)
def get(data, *fields):
    if not issubclass(type(data), dict): data=struct._from(data)._clonable()
    return struct(**data).get(*fields)

class table(struct):
    def _default(field, name, value):
        try: return getattr(field, name)
        except: setattr(field, name, value)
        return value
    def clear(this, *fields):
        sets = this.sets
        if not fields: fields = sets
        if fields:
            set = [field for field in fields if hasattr(this,field) and not ismethod(getattr(this, field))]
            for field in set: delattr(this, field)
    def has(this, *fields):
        return all([hasattr(this, field) for field in fields])
    def has_not(this, *fields): return not this.has(*fields)
    def check(this, **KV):
        try: check = [KV[key]==this.__dict__[key] for key in KV]
        except: return False
        return all(check)
    def find(this, _type):
        return [value for value in this.sets if class_of(get(this,value)[0]).inherits(_type)]
    def clone(this): 
        clone = type(this)()
        sets = this._clonable()
        clone.set(**sets)
        return clone

def isiterable(this): return isinstance(this, iterable) and type(this) is not str
def default(field, name, value): return table(**field)._default(name, value)

def ni(list):
    if isiterable(list):
        for n,i in enumerate(list): yield n,i
    else:
        for n,i in enumerate(list.__dict__.keys()): yield n,i

class at(table):
    DAY, HOUR, MIN = 86400, 3600, 60
    def __init__(_, dtime=None, **sets):
        _.set(**sets)
        if some(dtime) and issubclass(type(dtime), _time): _._time = dtime
        else:
            d,h,m,s,ms = _._default('d',0), _._default('h',0), _._default('m',0), _._default('s',0), _._default('ms',0)
            if not any([d,h,m,s,ms]): now=datetime.now(); _._time = now-datetime(now.year, now.month, now.day)
            else: _._time = _time(days=d, hours=h, minutes=m, seconds=s, milliseconds=ms)
        _.clear('d','h','m','s','ms')
    def __sub__(_, dtime):
        of=type(dtime); sets=_._clonable()
        if issubclass(of, _time): return at(_._time-dtime, **sets)
        elif issubclass(of, at): sets.update(dtime._clonable()); return at(_._time-dtime._time, **sets)
    def __add__(_, dtime):
        of=type(dtime); sets=_._clonable()
        if issubclass(of, _time): return at(_._time+dtime, **sets)
        elif issubclass(of, at): sets.update(dtime._clonable()); return at(_._time+dtime._time, **sets)
    def __str__(_): return str(_._time)
    @property
    def seconds(_): return _._time.seconds
    @property
    def S(_): return _.seconds
    @property
    def minutes(_): return _._time.seconds/60
    @property
    def M(_): return _.minutes
    @property
    def hours(_): return _.minutes/60
    @property
    def H(_): return _.hours
    @property
    def days(_): return _._time.days
    @property
    def D(_): return _.days
    @staticmethod
    def zero(): return at(_time())

from inspect import isfunction, ismethod, isgeneratorfunction, isgenerator, isroutine
from inspect import isabstract, isclass, ismodule, istraceback, isframe, iscode, isbuiltin
from inspect import ismethoddescriptor, isdatadescriptor, isgetsetdescriptor, ismemberdescriptor
from inspect import isawaitable, iscoroutinefunction, iscoroutine

from collections.abc import Iterable as iterable

import pickle

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
        with open(filename, 'rb') as file: return pickle.load(file)
    def save(data, filename):
        with open(filename, 'wb') as file: pickle.dump(data, file)
        
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

from numpy import ndarray, resize, linspace, arange
from numpy import min, max, average, floor
from numpy import ubyte, zeros, array
from scipy.signal import lfilter, butter
from matplotlib import pylab as lab

_NOTCH = _FR = 50
_SAMPLING = 500
_CONTINUOUS = 1
_UNIT = 'ms'

class rec(table, ndarray):
    @property
    def dimensions(of): return len(of.shape)
    @property
    def is_scalar(this): return this.shape is ()
    @property
    def is_vector(this): return len(this.shape)==1
    @property
    def is_matrix(this): return len(this.shape)>1
    @property
    def is_cube(this): return len(this.shape) == 3
    @property
    def is_binary(this): return this.dtype == ubyte and max(this) == 1
    @property
    def serialized(data):
        if not data.is_scalar and data.dimensions>1:
            return rec.read(data.T.flatten(), _deser=data.T.shape, **meta(data))
        return data
    @property
    def deserialized(data):
        if data.has('_deser'):
            deser = rec.read(resize(data, data._deser).T, **meta(data))
            deser.clear('_deser')
            return deser
        return data
    @property
    def as_matrix(data):    #implement numpy matrix
        if data.is_vector: return rec.read([data], to=type(data), **meta(data))
        return data
    @property
    def raw(data):
        if data.shape[0] == 1: return rec.read(data[0], **meta(data)).raw
        return data
    def join(base, *parts, **sets):
        flip, parts = None, list(parts)
        if 'flip' in sets: flip=sets.pop('flip')
        next = parts[0]
        if len(parts)>1: next = rec.join(parts[0], parts[1:])
        congruent = base.dimensions == next.dimensions and base.dimensions < 3
        if congruent:
            sets.update(base._clonable())
            A, B = base, next
            if flip: A, B = base.T, next.T
            C = record(A.tolist()+B.tolist(), **sets)
            if flip: return record(C.T, **sets)
            return C
    def get_as(this, data, cast=None):
        source = this
        if no(cast):
            if issubclass(type(data), rec): cast = type(data)
            else: cast = type(this)
        if issubclass(type(data), ndarray): source = resize(this, data.shape)
        return rec.read(source, to=cast, **meta(data))
    @staticmethod
    def read(iterable, to=None, **sets):
        if no(to) or not issubclass(to, rec): to = rec
        data = array(iterable).view(to)
        data.set(**sets)
        return data
    def clone(this, **sets):
        copy = this.copy().view(type(this))
        sets.update(this._clonable())
        copy.set(**sets)
        return copy
    def exclude(data, *items, **sets):
        if no(items) or data.is_scalar: return data
        excluded, items = None, [item for item in range(len(data)) if item not in items]
        if data.is_vector: excluded = rec.read([data])[:,items][0]
        else: excluded = data[items,:]
        return rec.read(excluded, to=type(data), **meta(data), **sets)
    def include(data, *items, **sets):
        if no(items) or data.is_scalar: return data
        included = []
        if data.is_vector: included = rec.read([data])[:,items][0]
        else: included = data[items,:]
        return rec.read(included, to=type(data), **meta(data), **sets)

create = record = rec.read
line = linspace

def series(ori,end=None,by=1):
    if no(end): end=ori; ori=0
    if not issubclass(type(by), int): return create(arange(ori,end,by))
    return array(range(ori,end,by))

def plot(data, at = 0., spacing = 1., color = 'k', width = 1., offset=0.): #review
    draw = record(data, **meta(data)); at = spacing*draw.as_matrix.shape[0]
    axes = lab.gca(); axes.set_ylim([at+max(data),0-max(data)]); at=0
    for n, row in ni(draw.as_matrix):
        if some(offset): row = draw[n]-average(row)+offset
        c, w = color, width
        if isiterable(color): c = color[n]
        if isiterable(width): w = width[n]
        lab.plot(at+row+n*spacing, color = c, linewidth = w)

def butter_type(lowcut, highcut, fs, order=5, type='band'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=type)
    return b, a

def butter_filter(data, lowcut, highcut, fs, order=5, type='band'):
    b, a = butter_type(lowcut, highcut, fs, order=order, type=type)
    y = lfilter(b, a, data)
    return y

def _to_rec(this):
    if not issubclass(type(this), rec): return rec.read(this, **meta(this)), rec
    return this, type(this)

def _prefilt(data, fs):
    pre = []
    for line in data.as_matrix:
        vector = line.tolist()
        pre.append(vector[:int(fs)]+vector)
    return record(pre)

def _postfilt(data, fs):
    post = []
    for line in data:
        vector = line.tolist()
        post.append(vector[int(fs):])
    return post

def notch(this, using=butter_type, fs=_SAMPLING, size=2, at=_NOTCH, order=5):
    data, type = _to_rec(this)
    if data.has('sampling'): fs=data.sampling
    nyq, cutoff = fs / 2., []
    for f in range(int(at), int(nyq), int(at)):
        cutoff.append((f - size, f + size))
    signal = _prefilt(data, fs)
    for bs in cutoff:
        low,hi = bs
        b,a = butter_type(low,hi,fs,order,'bandstop')
        signal = lfilter(b,a,signal)
    return record(_postfilt(signal, fs), to=type, **meta(data))

def band(this, low_high, fs=_SAMPLING, using=butter_filter, order=5):
    data, type = _to_rec(this)
    if data.has('sampling'): fs=data.sampling
    low, high = min(low_high), max(low_high)
    if low<1.: low = 1.
    tailed = _prefilt(data, fs)
    tailed = using(tailed, low, high, fs, order)
    return rec.read(_postfilt(tailed, fs), to=type, **meta(data))

def binarize(this):
    data, type = _to_rec(this)
    if data.is_binary: return data
    rows = []
    for row in data.as_matrix:
        d = row - array([row[-1]]+row[:-1].tolist())
        d[d>=0] = 1; d[d<0] = 0
        rows.append(d.astype(ubyte))
    return rec.read(rows, to=type).get_as(data)

def halve(matrix):
    halved, (data, type) = [], _to_rec(matrix)
    for line in data.as_matrix:
        h = resize(line, (int(len(line)/2), 2))
        halved.append((h[:,0]+h[:,1])/2.)
    return rec.read(halved, to=type, **meta(data))

def dwindle(matrix, by=1):
    if by: return dwindle(halve(matrix), by-1)
    return matrix

def upsample(matrix, fs1, fs2):
    y=zeros((matrix.shape[0],fs2))
    if fs1 < fs2:
        #upsampling by a factor R
        L=matrix.shape[1]
        R=int(floor(fs2/fs1)+1)
        for i,e in enumerate(matrix):
            ups=[]
            for j in range(L-1):
                if j>0: ups.append(list(linspace(matrix[i][j],matrix[i][j+1],R)[1:3]))
                else: ups.append(list(linspace(matrix[i][j],matrix[i][j+1],R)[0:3]))
            for k,s in enumerate(sum(ups, [])): y[i][k]=s 
            y[i][-1]=y[i][-2]
        return rec.read(y)
    else: print("Error: fs1 >= fs2")

def remap(this, axis=None, base=0, top=1., e=0):
    def map(x, b, t, e): return ((x-min(x)+e)/(max(x)-min(x)+e)+b)*(t-b)
    data, type = _to_rec(this)
    if no(axis): return rec.read(map(this, base, top, e), to=type, **meta(data))
    rows = data.as_matrix
    if axis==0 or axis>1: rows = rows.T
    remapped = []
    for row in rows: remapped.append(map(row, base, top, e))
    if axis==0 or axis>1: rows = rows.T
    return rec.read(remapped).get_as(data)

from numpy import correlate, average, array, angle, mean, sign, exp, zeros, abs, unwrap, fromfile, unpackbits, packbits
from scipy.signal import hilbert, csd, lfilter, butter

def phaselock(signal1, signal2):
    """Computes the phase locking value between two notch-filtered signals.
    
    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.

    Returns:
        float: Phase locking value.
    """
    sig1_hil = hilbert(signal1)                          
    sig2_hil = hilbert(signal2)
    phase1 = angle(sig1_hil)                           
    phase2 = angle(sig2_hil)
    phase_dif = phase1-phase2                             
    plv = abs(mean(exp(complex(0,1)*phase_dif)))    
    return plv

def phaselag(signal1, signal2):
    """Computes the phase lag index between two signals.
    
    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.

    Returns:
        float: Phase lag index.
    """
    sig1_hil = hilbert(signal1)                 
    sig2_hil = hilbert(signal2)
    phase1 = angle(sig1_hil)                 
    phase2 = angle(sig2_hil)
    phase_dif = phase1-phase2                   
    pli = abs(mean(sign(phase_dif)))     
    return pli

def spectral_coherence(signal1, signal2, fs, imag=False):
    """Computes the spectral coherence between two signals.

    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.
        fs (int): Sampling frequency.
        imag (bool): If True, computed the imaginary part of spectral coherence, if False computes the real part. Defaults to False.

    Returns:
        float: Spectral coherence.
    """
    Pxy = csd(signal1,signal2,fs=fs, scaling='spectrum')[1] 
    Pxx = csd(signal1,signal1,fs=fs, scaling='spectrum')[1]
    Pyy = csd(signal2,signal2,fs=fs, scaling='spectrum')[1]
    if imag: return average((Pxy.imag)**2/(Pxx*Pyy))     
    elif not imag: return average(abs(Pxy)**2/(Pxx*Pyy))

def cross_correlation(signal1, signal2):
    """Computes the cross correlation between two signals.
    
    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.

    Returns:
        float: Cross correlation.
    """
    return correlate(signal1, signal2, mode="valid")

def PAC(signal1, signal2, fs):
    """Computes low frequency phase-high frequency amplitude phase coupling between two signals.
    Low frequency = 1-4 Hz; High frequency = 30-70 Hz
    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.
        fs (int): Sampling frequency.

    Returns:
        float: Phase-amplitude coupling.
    """   
    low = butter_filter(signal1,1,4,fs) 
    high = butter_filter(signal2,30,70,fs) 
    low_hil = hilbert(low)
    low_phase_angle = unwrap(angle(low_hil))   
    high_env_hil = hilbert(abs(hilbert(high)))
    high_phase_angle = unwrap(angle(high_env_hil))
    phase_dif = low_phase_angle - high_phase_angle 
    plv = abs(mean(exp(complex(0,1)*phase_dif)))
    return plv

class bit:
    def __init__(my, size = 32): my.states = size
    def resize(this, bits):
        n, max = 0, 1
        for bit in bits:
            n += bit * max
            max <<= 1
        n = int(round(n / float(max) * this.states)) - 1
        max, bits = 1, []
        while(max < this.states):
            bits.append((n & max) / max)
            max <<= 1
        return bits

class AWC(struct, dict):
    train, limit = 10, 100
    bits = 8
    time = 8
    blur = None
    same = False
    class lnx:
        l, n = 1, 2
        @property
        def _clone(this):
            copy = AWC.lnx()
            copy.l, copy.n = this.l, this.n
            return copy
        def __add__(this, bit):
            bit, learn, limit = bit
            this.l += bit * learn
            this.n += learn
            if this.n > limit: this.n /= 2.; this.l /= 2.
        def __call__(set, weight): return set.l * weight, set.n * weight
    def __init__(context, **params):
        context._set(_codes=[], _last=None, **params)
    def clone(this, **changes):
        copy = AWC()
        copy.set(**this._clonable)
        copy.set(**changes)
        copy.update(this)
        for context in copy: copy[context] = copy[context]._clone
        return copy
    def _make(actual, set):
        actual._codes, context = [], []
        for bits in set:
            context += bits
            actual._codes.append(tuple(context))
    def __call__(actual):
        l, n, w = 1., 2., 1
        for length, context in enumerate(actual._codes):
            w *= len(context)
            if context in actual:
                _l, _n = actual[context](w)
                l += _l; n += _n
            else: actual[context] = AWC.lnx()
        return l/n
    def __add__(last, bit):
        for context in last._codes: last[context] + bit
    def learn(symbol, data=None, file=None, tell=False):
        if file: data = unpackbits(fromfile(file, dtype = 'ubyte'))
        check, to = None, 0.
        if tell: check = tell*len(data)
        train, limit = symbol.train, symbol.limit
        set, coded, time = [], [], symbol.time
        while time: set.append(list()); time -= 1
        for n, bit in enumerate(data.tolist()):
            if tell and n%check==0: print('{:.0%}|'.format(to), end=''); to+=tell
            if symbol.same: symbol._make([[n%symbol.bits]]+set[1:])
            else: symbol._make(set)
            coded.append(symbol())
            symbol + (bit, train, limit)
            base = set[0]
            if len(base) == symbol.bits:
                if symbol.blur: base = symbol.blur.resize(base)
                set.insert(1, packbits(base).tolist())
                set.pop(-1)
                set[0] = [bit]
            else: base.append(bit)
        symbol._last = data
        return dict(code=record(coded, **meta(data)), error=record(abs(coded-data), **meta(data)))

def error(matrix, layers=1, mmult=3, tell=.1, dtail=True):
    d = len(matrix)
    E, pairs = zeros((d,d)), []
    O, l = zeros((d,d)), zeros((d,d))
    for a in range(d):
        if dtail:
            for b in range(d): pairs.append((a,b))
        else:
            for b in range(a, d): pairs.append((a,b))        
    to, check = None, None
    if tell: to, check = 0., int(tell*len(pairs))
    for n, (a, b) in enumerate(pairs):
        if tell=='all': print('.', end='')
        elif tell<1. and n%check==0: print('{:.0%}|'.format(to), end=''); to+=tell
        c, data = AWC(bits=2, time=2*mmult), create([matrix[a],matrix[b]])
        R = c.learn(binarize(data).serialized)
        if layers==1 or layers==3:
            E[b,a] += average(R['error'].deserialized[0])
            E[a,b] += average(R['error'].deserialized[1])
        if layers>1:
            O[a,b] = average(R['error'].deserialized[0])
            l[a,b] = average(R['error'].deserialized[1])
        del c,R
    if tell and tell!='all' and tell<1.: print()
    elif tell == 1.: print('.', end='')
    if layers==1: return record(E)
    elif layers==2: return record(O), record(l)
    return record(E), record(O), record(l)

def PEC(nse, n):
    """Computes prediction error connectivity.

    Args:
        nse (list): Preprocessed epoch (resampled and notched).
        n (int): Epoch index.

    Returns:
        array: Connectivity matrix.
    """
    print('{}: '.format(n), end='')
    return array(error(nse, 2)[1]) 

from joblib import Parallel, delayed

def analyze_epoch(epoch, method, dtail, **opts):
    mat = zeros((len(epoch),len(epoch)))                                    
    nid, pairs = list(range(len(epoch))), []
    for a in range(len(nid)):                             
        if dtail:
            for b in range(len(nid)): pairs.append((a,b))
        else:
            for b in range(a, len(nid)): pairs.append((a,b))                                       

    parallelize = Parallel(n_jobs=-1)(delayed(method)(epoch[pair[0]], epoch[pair[1]], **opts) for pair in pairs)
    conn_per_pair = [p for p in parallelize]

    for pair_idx, pair in enumerate(pairs):                                                       
        mat[pair[0],pair[1]] = conn_per_pair[pair_idx]
    return mat

def connectivity_analysis(epochs, method, dtail=True, **opts):
    """Computes a connectivity matrix N×N (N - number of nodes) per epoch, containing connectivity method measures for all node pairs.

    Args:
        epochs (list): List of preprocessed epochs (resampled, filtered and notched).
        method (function): Connectivity method.
        dtail (bool): If True, computes a square matrix; if False, computes a tringular matrix. Defaults to False.
        opts (optional): method-specific arguments.
    Returns:
        list: List of connectivity matrices for all epochs.
    """
    print('Connectivity measure: '+str(method).split()[1])
    parallelize = Parallel(n_jobs=-1)(delayed(analyze_epoch)(e, method, dtail, **opts) for e in epochs)
    cm = [mat for mat in parallelize]
    return cm

from numpy import delete

def exclude_node_from_cm(cm_list, channel_id):
    """Excludes a node from connectivity matrices of cm_list epochs.
    Arguments:
        cm_list (list of numpy arrays): list of connectivity matrices saved in the PREP file under X;
                                        First half of the matrices are from seizure epochs, second half form baseline epochs.
        channel_id (int or list): index of the node to be removed
    Returns the updated list of matrices.
    """
    reduced_cms = []
    for cm in cm_list:
      cm = delete(cm, channel_id, 0)
      cm = delete(cm, channel_id, 1)
      reduced_cms.append(cm)
    return reduced_cms

import time

_start_time = time.time()

def timer_start():
    global _start_time 
    _start_time = time.time()

def timer_end():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))

# --------------------------------------------------------------------------------------------------------------------------------------------------

from sys import argv
from os import getcwd, makedirs

main_folder = getcwd()

file_path = argv[1]
print(file_path)

woi_code = {'1':"baseline", '2':"preseizure5", '3':"preseizure4", '4':"preseizure3", '5':"preseizure2", '6':"preseizure1", '7':"transition1", '8':"transition2", '9':"transition60", '10':"seizure"}
woi_code_inv = dict((v,k) for k, v in woi_code.items())

filename = file_path.split("/")[-1]
subject_id = filename.split("-")[0]
woi = woi_code_inv[filename.split("-")[1]] if len(filename.split("-"))==3 else woi_code_inv[filename.split("-")[-1].split(".")[0]]
bands = filename.split("-")[2].split(".")[0] if len(filename.split("-"))==3 else None

span, step = 1000, 500      # in ms
min_woi_duration = 60000    # in ms
n_epochs = int(((min_woi_duration/step)-1))

# original nodes per sub
nodes = {
"1":['P1-P2', 'P4-P5', 'P8-P9', 'P9-P10', 'P10-P11', 'G1-G2', 'G8-G9', 'G9-G10', 'G10-G11', 'G11-G12', 'M1-M2', 'M8-M9', 'M9-M10', 'M10-M11', 'M11-M12', 'O1-O2', 'O2-O3', 'O5-O6', 'O6-O7', 'F1-F2', 'F7-F8', 'F8-F9', 'F9-F10', 'F10-F11', 'F11-F12', 'F12-F13', 'A1-A2', 'A2-A3', 'A3-A4', 'A7-A8', 'A8-A9', 'A9-A10', 'A10-A11', 'B1-B2', 'B2-B3', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'C1-C2', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'Q1-Q2', 'Q2-Q3', 'Q3-Q4', 'Q4-Q5', 'Q8-Q9', 'Q9-Q10', 'Q10-Q11', 'Q11-Q12', 'T1-T2', 'T2-T3', 'T3-T4', 'T4-T5', 'T5-T6', 'T6-T7', 'T7-T8', 'T8-T9', 'T9-T10', 'T10-T11', 'T11-T12', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'D5-D6', 'D6-D7', 'D7-D8', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E5-E6', 'E6-E7', 'E7-E8', 'E8-E9', 'E9-E10', 'E10-E11', 'L1-L2', 'L2-L3', 'L5-L6', 'L6-L7', 'L7-L8', 'U1-U2', 'U2-U3', 'U3-U4', 'U4-U5', 'U5-U6', 'U6-U7', 'J1-J2', 'J9-J10', 'J10-J11', 'J11-J12', 'J12-J13', 'J13-J14', 'J14-J15'],
"2":['A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'B9-B10', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'C10-C11', 'C11-C12', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E5-E6', 'E6-E7', 'E7-E8', 'F1-F2', 'F2-F3', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F7-F8'],
"3":["T'1-T'2", "T'2-T'3", "T'3-T'4", "T'4-T'5", "T'5-T'6", "T'6-T'7", "T'7-T'8", "A'1-A'2", "A'2-A'3", "A'3-A'4", "A'4-A'5", "A'5-A'6", "A'6-A'7", "A'7-A'8", "A'8-A'9", "A'9-A'10", "B'1-B'2", "B'2-B'3", "B'3-B'4", "B'4-B'5", "B'5-B'6", "B'6-B'7", "B'7-B'8", "B'8-B'9", "B'9-B'10", "C'1-C'2", "C'2-C'3", "C'3-C'4", "C'4-C'5", "C'5-C'6", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10", "C'10-C'11", "C'11-C'12", "E'1-E'2", "E'2-E'3", "E'3-E'4", "E'4-E'5", "E'5-E'6", "E'6-E'7", "E'7-E'8", "E'8-E'9", "E'9-E'10", "D'1-D'2", "D'2-D'3", "D'3-D'4", "D'4-D'5", "D'5-D'6", "D'6-D'7", "D'7-D'8", "D'8-D'9", "D'9-D'10", "D'10-D'11", "D'11-D'12", "W'1-W'2", "W'2-W'3", "W'3-W'4", "W'4-W'5", "W'5-W'6", "W'6-W'7", "W'7-W'8", "W'8-W'9", "W'9-W'10", "W'10-W'11", "W'11-W'12", "W'12-W'13", "W'13-W'14", "W'14-W'15", "K'1-K'2", "K'2-K'3", "K'3-K'4", "K'4-K'5", "K'5-K'6", "K'6-K'7", "K'7-K'8", "K'8-K'9", "K'9-K'10", "K'10-K'11", "K'11-K'12", "K'12-K'13", "K'13-K'14", "K'14-K'15", "P'1-P'2", "P'2-P'3", "P'3-P'4", "P'4-P'5", "P'5-P'6", "P'6-P'7", "P'7-P'8", "P'8-P'9", "P'9-P'10", "P'10-P'11", "P'11-P'12", "P'12-P'13", "P'13-P'14", "P'14-P'15", "L'1-L'2", "L'2-L'3", "L'3-L'4", "L'4-L'5", "L'5-L'6", "L'6-L'7", "L'7-L'8", "L'8-L'9", "L'9-L'10", "L'10-L'11", "L'11-L'12", "O'1-O'2", "O'2-O'3", "O'3-O'4", "O'4-O'5", "O'5-O'6", "O'6-O'7", "O'7-O'8", "O'8-O'9", "O'9-O'10", "O'10-O'11", "O'11-O'12", "X'1-X'2", "X'2-X'3", "X'3-X'4", "X'4-X'5", "X'5-X'6", "X'6-X'7", "X'7-X'8", "X'8-X'9", "X'9-X'10", "X'10-X'11", "X'11-X'12", "X'12-X'13", "X'13-X'14", "X'14-X'15"],
"4":['A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A6-A7', 'A7-A8', 'A8-A9', 'A9-A10', 'A10-A11', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B11-B12', 'C1-C2', 'C2-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'C10-C11', 'C11-C12', 'R1-R2', 'R2-R3', 'R3-R4', 'R4-R5', 'R5-R6', 'R6-R7', 'R7-R8', 'R8-R9', 'R9-R10', 'R10-R11', 'R11-R12', 'R12-R13', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-L6', 'L6-L7', 'L7-L8', 'L8-L9', 'L9-L10', 'O1-O2', 'O2-O3', 'O3-O4', 'O4-O5', 'O5-O6', 'O6-O7', 'O7-O8', 'Q1-Q2', 'Q2-Q3', 'Q3-Q4', 'Q4-Q5', 'Q5-Q6', 'Q6-Q7', 'Q7-Q8', 'Q8-Q9', 'Q9-Q10', 'T1-T2', 'T2-T3', 'T3-T4', 'T4-T5', 'T5-T6', 'T6-T7', 'T7-T8', 'T8-T9', 'T9-T10', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D6', 'D6-D7', 'D7-D8', 'D8-D9', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E5-E6', 'E6-E8', 'E8-E9', 'E9-E10', 'E10-E11', 'J1-J2', 'J2-J3', 'J6-J7', 'J7-J8', 'J8-J9', 'J9-J10', 'J10-J11', 'J11-J12', 'I1-I2', 'I2-I3', 'I3-I4', 'I4-I5', 'I5-I6', 'I6-I7', 'I7-I8', 'P4-P5', 'P5-P6', 'P6-P7', 'P7-P8', 'P8-P9', 'P9-P10'],
"5":["A'1-A'2", "A'2-A'3", "A'3-A'4", "A'4-A'5", "A'5-A'6", "A'6-A'7", "A'7-A'8", "A'8-A'9", "A'9-A'10", "B'1-B'2", "B'2-B'3", "B'3-B'4", "B'4-B'5", "B'5-B'6", "B'6-B'7", "B'7-B'8", "B'8-B'9", "B'9-B'10", "B'10-B'11", "B'11-B'12", "C'1-C'2", "C'2-C'3", "C'3-C'4", "C'4-C'5", "C'5-C'6", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10", "C'10-C'11", "C'11-C'12", "D'1-D'2", "D'2-D'3", "D'3-D'4", "D'4-D'5", "D'5-D'6", "D'6-D'7", "D'7-D'8", "F'1-F'2", "F'2-F'3", "F'3-F'4", "F'4-F'5", "F'5-F'6", "F'6-F'7", "F'7-F'8", "H'1-H'2", "H'2-H'3", "H'3-H'4", "H'4-H'5", "G'1-G'2", "G'2-G'3", "G'3-G'4", "G'4-G'5", "G'5-G'6", "G'6-G'7", "G'7-G'8", "G'8-G'9", "G'9-G'10", "G'10-G'11", "G'11-G'12"],
"6":['A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A6-A7', 'A7-A8', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'B9-B10', 'B10-B11', 'B11-B12', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'C10-C11', 'C11-C12', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'D5-D6', 'D6-D7', 'D7-D8', 'D8-D9', 'D9-D10', 'D10-D11', 'D11-D12', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E5-E6', 'E6-E7', 'E7-E8', 'E8-E9', 'E9-E10', 'E10-E11', 'E11-E12', 'F1-F2', 'F2-F3', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F7-F8', 'F8-F9', 'F9-F10', 'F10-F11', 'F11-F12', 'G1-G2', 'G2-G3', 'G3-G4', 'G4-G5', 'H1-H2', 'H2-H3', 'H3-H4', 'H4-H5', 'H5-H6', 'H6-H7', 'H7-H8', 'H8-H9', 'H9-H10', 'H10-H11', 'H11-H12', 'I1-I2', 'I2-I3', 'I3-I4', 'I4-I5', 'I5-I6', 'I6-I7', 'I7-I8', 'I8-I9', 'I9-I10', 'I10-I11', 'I11-I12', "C'1-C'2", "C'2-C'3", "C'3-C'4", "C'4-C'5", "C'5-C'6", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10"],
"7":['A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A6-A7', 'A7-A8', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'B9-B10', 'B10-B11', 'B11-B12', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', "C'1-C'2", "C'2-C'3", "C'3-C'4", "C'4-C'5", "C'5-C'6", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10", "C'10-C'11", "C'11-C'12", 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'D5-D6', 'D6-D7', 'D7-D8', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E5-E6', 'E6-E7', 'E7-E8', 'E8-E9', 'E9-E10', 'F1-F2', 'F2-F3', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F7-F8', 'F8-F9', 'F9-F10', 'F10-F11', 'F11-F12', 'G1-G2', 'G2-G3', 'G3-G4', 'G4-G5', 'G5-G6', 'G6-G7', 'G7-G8', 'G8-G9', 'G9-G10', 'H1-H2', 'H2-H3', 'H3-H4', 'H4-H5', 'H5-H6', 'H6-H7', 'H7-H8', 'I1-I2', 'I2-I3', 'I3-I4', 'I4-I5', 'I5-I6', 'I6-I7', 'I7-I8', 'I8-I9', 'I9-I10', 'I10-I11', 'I11-I12', 'J1-J2', 'J2-J3', 'J3-J4', 'J4-J5', 'J5-J6', 'J6-J7', 'J7-J8'],
"8":["F'1-F'2", "F'2-F'3", "F'8-F'9", "F'9-F'10", "F'10-F'11", "F'11-F'12", "T'1-T'2", "T'2-T'3", "T'3-T'4", "T'4-T'5", "T'5-T'6", "T'6-T'7", "T'7-T'8", "T'8-T'9", "T'9-T'10", "A'1-A'2", "A'2-A'3", "A'3-A'4", "A'4-A'5", "A'5-A'6", "A'6-A'7", "A'7-A'8", "A'8-A'9", "A'9-A'10", "A'10-A'11", "A'11-A'12", "B'1-B'2", "B'2-B'3", "B'3-B'4", "B'4-B'5", "B'8-B'9", "B'9-B'10", "B'10-B'11", "B'11-B'12", "C'1-C'2", "C'2-C'3", "C'3-C'4", "C'4-C'5", "C'5-C'6", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10", "C'10-C'11", "D'1-D'2", "D'2-D'3", "D'3-D'4", "D'4-D'5", "D'5-D'6", "D'6-D'7", "D'7-D'8", "D'8-D'9", "D'9-D'10", "S'1-S'2", "S'2-S'3", "S'5-S'6", "S'6-S'7", "S'7-S'8", "S'8-S'9", "S'9-S'10", "S'10-S'11", "S'11-S'12", "S'12-S'13", "S'13-S'14", "I'1-I'2", "I'2-I'3", "I'3-I'4", "I'4-I'5", "I'5-I'6", "I'6-I'7", "I'7-I'8", "I'8-I'9", "I'9-I'10", "W'1-W'2", "W'2-W'3", "W'10-W'11", "W'11-W'12", "W'12-W'13", "W'13-W'14", "W'14-W'15", "W'15-W'16", "W'16-W'17", "W'17-W'18", "U'1-U'2", "U'2-U'3", "U'3-U'4", "U'4-U'5", "U'8-U'9", "U'9-U'10", "U'10-U'11", "U'11-U'12", "U'12-U'13", "U'13-U'14", "U'14-U'15", "O'1-O'2", "O'2-O'3", "O'3-O'4", "O'4-O'5", "O'5-O'6", "O'6-O'7", "O'7-O'8", "O'8-O'9", "O'9-O'10", "O'10-O'11", "O'11-O'12", "O'12-O'13", "O'13-O'14", "O'14-O'15"],
"9":['FC1-FC2', 'FC2-FC3', 'FC3-FC4', 'FC4-FC5', 'FC5-FC6', 'FC6-FC7', 'FC7-FC8', 'FC8-FC9', 'FC9-FC10', 'A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A6-A7', 'A7-A8', 'A8-A9', 'A9-A10', 'HAn1-HAn2', 'HAn2-HAnt3', 'HAnt3-HAnt4', 'HAnt4-HAnt5', 'HAnt5-HAnt6', 'HAnt6-HAnt7', 'HAnt7-HAnt8', 'HAnt8-HAnt9', 'HAnt9-HAnt10', 'HAnt10-Ref', 'HAnt11-Ref', 'HP1-HP2', 'HP2-HP3', 'HP3-HP4', 'HP4-HP5', 'HP5-HP6', 'HP6-HP7', 'HP7-HP8', 'HP8-HP9', 'HP9-HP10', 'TB1-TB2', 'TB2-TB3', 'TB3-TB4', 'TB4-TB5', 'TB5-TB6', 'TB6-TB7', 'TB7-TB8', 'TB8-TB9', 'TB9-TB10', 'TB10-TB11', 'TB11-TB12'],
"10":['F1-F2', 'F2-F3', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F7-F8', 'F8-F9', 'F9-F10', 'F10-F11', 'F11-F12', 'F12-F13', 'F13-F14', 'F14-F15', 'F15-F16', 'F16-F17', 'A1-A3', 'A3-A5', 'A5-A7', 'A7-A9', 'A9-A11', 'A11-A13', 'A13-A15', 'B1-B3', 'B3-B5', 'B5-B7', 'B7-B9', 'C1-C3', 'C3-C5', 'C5-C7', 'C7-C9', 'C9-C11', 'D1-D3', 'D3-D5', 'D5-D7', 'D7-D9', 'D9-D11', 'D11-D13', 'D13-D15', 'E1-E3', 'E3-E5', 'E5-E7', 'E7-E9', 'E9-E11', 'J1-J3', 'J3-J5', 'J5-J7', 'J7-J9', 'J9-J11', 'J11-J13', 'J13-J15', 'J15-J17', 'K1-K3', 'K3-K5', 'K5-K7', 'K7-K9', 'K9-K11', 'K11-K13', 'K13-K15', 'L1-L3', 'L3-L5', 'L5-L7', 'L7-L9', 'L9-L11', 'L11-L13', 'L13-L15', 'M1-M3', 'M3-M5', 'M5-M7', 'M7-M9', 'O1-O3', 'O3-O5', 'O5-O7', 'P1-P3', 'P3-P5', 'P5-P7', 'P7-P9', 'R1-R3', 'R3-R5', 'R5-R7', 'R7-R9', 'R9-R11', 'R11-R13', 'R13-R15', 'S1-S3', 'S3-S5', 'S5-S7', 'S7-S9', 'S9-S11', 'S11-S13', 'S13-S15', 'T1-T3', 'T3-T5', 'T5-T9', 'T9-T11'],
"11":["R'1-R'2", "R'2-R'3", "R'9-R'10", "R'10-R'11", "R'11-R'12", "R'12-R'13", "R'13-R'14", "S'1-S'2", "S'2-S'3", "S'6-S'7", "P'1-P'2", "P'2-P'3", "P'3-P'4", "P'4-P'5", "P'7-P'8", "P'8-P'9", "M'1-M'2", "M'8-M'9", "M'9-M'10", "M'10-M'11", "M'13-M'14", "J'1-J'2", "J'7-J'8", "J'8-J'9", "J'9-J'10", "K'1-K'2", "K'2-K'3", "K'7-K'8", "K'8-K'9", "K'9-K'10", "L'2-L'3", "L'3-L'4", "L'4-L'5", "L'5-L'6", "B'1-B'2", "B'5-B'6", "B'6-B'7", "B'7-B'8", "B'8-B'9", "C'1-C'2", "C'2-C'3", "C'9-C'10", "C'10-C'11", "C'11-C'12", "O'4-O'5", "O'5-O'6", "O'6-O'7", "O'10-O'11", "O'11-O'12", "O'12-O'13", "Q'3-Q'4", "Q'4-Q'5", "Q'5-Q'6", "Q'6-Q'7", "Q'10-Q'11", "Q'11-Q'12", 'S1-S2', 'S8-S9', 'S9-S10', 'S10-S11', 'P1-P2', 'P2-P3', 'P6-P7', 'P7-P8', 'K2-K3', 'K3-K4', 'K4-K5', 'L1-L2', 'L2-L3', 'L3-L4', 'L7-L8', 'J1-J2', 'J11-J12', 'J12-J13', 'J13-J14', 'C1-C2', 'C2-C3', 'C8-C9', 'C9-C10', 'Q1-Q2', 'Q2-Q3', 'Q3-Q4', 'Q12-Q13', 'Q13-Q14', 'O1-O2', 'O2-O3', 'O3-O4', 'O4-O5', 'O9-O10'],
"12":["E'1-E'2", "E'2-E'3", "E'3-E'4", "E'4-E'5", "E'5-E'6", "E'6-E'7", "E'7-E'8", "E'10-E'11", "O'1-O'2", "O'2-O'3", "O'3-O'4", "O'4-O'5", "O'5-O'6", "O'6-O'7", "O'7-O'8", "O'8-O'9", "O'9-O'10", "O'10-O'11", "C'1-C'2", "C'2-C'3", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10", "C'10-C'11", "T'1-T'2", "T'2-T'3", "T'8-T'9", "T'9-T'10", "T'10-T'11", "U'1-U'2", "U'4-U'5", "U'5-U'6", "U'8-U'9", "Q'1-Q'2", "Q'10-Q'11", "Q'11-Q'12", "Q'12-Q'13", "Q'13-Q'14", "Q'14-Q'15", "W'1-W'2", "W'6-W'7", "W'7-W'8", "W'8-W'9", "W'12-W'13", "W'13-W'14", "S'1-S'2", "S'8-S'9", "S'9-S'10", "S'10-S'11", "S'11-S'12", "P'1-P'2", "P'2-P'3", "P'3-P'4", "P'4-P'5", "P'5-P'6", "P'6-P'7", "P'7-P'8", "P'8-P'9", 'T1-T2', 'T2-T3', 'T3-T4', 'T4-T5', 'T5-T6', 'T11-T12', 'O1-O2', 'O2-O3', 'O5-O6', 'O6-O7', 'O7-O8', 'O8-O9', 'O9-O10', 'O10-O11', 'U1-U2', 'U5-U6', 'U6-U7', 'W1-W2', 'W2-W3', 'W3-W4', 'W9-W10', 'W10-W11', 'W11-W12', 'W12-W13', 'Q1-Q2', 'Q2-Q3', 'Q6-Q7', 'Q7-Q8', 'Q13-Q14', 'Q14-Q15', 'P1-P2', 'P4-P5', 'P5-P6', 'P6-P7', 'P10-P11'],
"13":['A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A6-A7', 'A7-A8', 'A8-A9', 'A9-A10', 'A10-A11', 'A11-A12', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'B9-B10', 'B10-B11', 'B11-B12', 'B12-B13', 'B13-B14', 'B14-B15', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'C10-C11', 'C11-C12', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'F1-F2', 'F2-F3', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F7-F8', 'F8-F9', 'F9-F10', 'F10-F11', 'F11-F12', 'G1-G2', 'G2-G3', 'G3-G4', 'G4-G5', 'G5-G6', 'G6-G7', 'G7-G8', 'H1-H2', 'H2-H3', 'H3-H4', 'H4-H5', 'H5-H6', 'H6-H7', 'H7-H8', 'H8-H9', 'H9-H10', 'H10-H11', 'H11-H12', 'I1-I2', 'I2-I3', 'I3-I4', 'I4-I5', 'I5-I6', 'I6-I7', 'I7-I8', 'I8-I9', 'I9-I10', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-L6', 'L6-L7', 'L7-L8'],
"14":['A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A6-A7', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'B9-B10', 'B10-B11', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'D5-D6', 'D6-D7', 'D7-D8', 'D8-D9', 'D9-D10', 'D10-D11', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E5-E6', 'E6-E7', 'E7-E8', 'E8-E9', 'F1-F2', 'F2-F3', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F7-F8', 'F8-F9', 'F9-F10', 'G1-G2', 'G2-G3', 'G3-G4', 'G4-G5', 'H1-H2', 'H2-H3', 'H3-H4', 'H4-H5', 'H5-H6', 'H6-H7', 'H7-H8', "G'1-G'2", "G'2-G'3", "G'3-G'4", "G'4-G'5", "C'1-C'2", "C'2-C'3", "C'3-C'4", "C'4-C'5", "C'5-C'6", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10"],
"15":["A'1-A'2", "A'2-A'3", "A'6-A'7", "A'7-A'8", "B'1-B'2", "B'2-B'3", "B'3-B'4", "B'4-B'5", "B'5-B'6", "B'6-B'7", "B'7-B'8", "C'1-C'2", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10", "E'1-E'2", "E'2-E'3", "E'5-E'6", "E'6-E'7", "E'7-E'8", "E'8-E'9", "T'1-T'2", "T'2-T'3", "T'3-T'4", "T'8-T'9", "T'9-T'10", "T'10-T'11", "T'11-T'12", "T'12-T'13", "T'13-T'14", "T'14-T'15", "J'1-J'2", "J'2-J'3", "J'3-J'4", "J'4-J'5", "J'12-J'13", "J'13-J'14", "J'14-J'15", "I'1-I'2", "I'2-I'3", "I'3-I'4", "I'8-I'9", "I'9-I'10", "I'10-I'11", "O'1-O'2", "O'6-O'7", "O'7-O'8", "O'8-O'9", "O'9-O'10", "O'10-O'11", "G'1-G'2", "G'2-G'3", "G'10-G'11", "G'11-G'12", "G'12-G'13", "G'13-G'14", "G'14-G'15", "Q'1-Q'2", "Q'6-Q'7", "Q'7-Q'8", "Q'8-Q'9", "Q'11-Q'12", "Q'12-Q'13", "P'1-P'2", "P'2-P'3", "P'3-P'4", "P'4-P'5", "P'5-P'6", "P'6-P'7", "P'7-P'8", "P'8-P'9", "P'9-P'10", "U'1-U'2", "U'5-U'6", "U'6-U'7", "U'7-U'8", "U'8-U'9", "M'1-M'2", "M'2-M'3", "M'3-M'4", "M'4-M'5", "M'7-M'8", "M'8-M'9", "M'9-M'10", "M'10-M'11", "M'11-M'12", 'F1-F2', 'F7-F8', 'F8-F9', 'F9-F10', 'F10-F11', 'M1-M2', 'M2-M3', 'M3-M4', 'M12-M13', 'M13-M14'],
"16":["F'1-F'2", "F'5-F'6", "F'8-F'9", "F'9-F'10", "M'1-M'2", "M'8-M'9", "M'9-M'10", "M'10-M'11", "M'11-M'12", "M'12-M'13", "M'13-M'14", "O'4-O'5", "O'5-O'6", "O'6-O'7", "Y'1-Y'2", "Y'2-Y'3", "Y'3-Y'4", "Y'4-Y'5", "Y'5-Y'6", "Y'6-Y'7", "Y'13-Y'14", "Y'14-Y'15", "Y'15-Y'16", "Y'16-Y'17", "I'1-I'2", "I'5-I'6", "I'6-I'7", "I'7-I'8", "U'1-U'2", "U'2-U'3", "U'3-U'4", "U'4-U'5", "U'5-U'6", "A'1-A'2", "A'7-A'8", "A'8-A'9", "A'9-A'10", "A'10-A'11", "B'1-B'2", "B'2-B'3", "B'8-B'9", "B'9-B'10", "B'10-B'11", "C'1-C'2", "C'2-C'3", "C'5-C'6", "C'6-C'7", "C'7-C'8", "C'10-C'11", "C'11-C'12", "D'1-D'2", "D'6-D'7", "D'7-D'8", "D'8-D'9", "E'2-E'3", "E'3-E'4", "E'4-E'5", "E'5-E'6", "E'6-E'7", "E'7-E'8", "E'8-E'9", "E'9-E'10", "E'10-E'11", "J'1-J'2", "J'8-J'9", "J'9-J'10", "J'10-J'11", "J'11-J'12", "J'12-J'13", "J'13-J'14", "J'14-J'15", "W'1-W'2", "W'11-W'12", "W'12-W'13", "W'13-W'14", "Q'1-Q'2", "Q'2-Q'3", "Q'6-Q'7", "Q'7-Q'8", "Q'8-Q'9", "Q'9-Q'10", "Q'10-Q'11", 'A1-A2', 'A2-A3', 'A8-A9', 'A9-A10', 'A10-A11', 'B1-B2', 'B7-B8', 'B8-B9', 'C1-C2', 'C7-C8', 'C8-C9'],
"17":['A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'B9-B10', 'B10-B11', 'B11-B12', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'C10-C11', 'C11-C12', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'D5-D6', 'D6-D7', 'D7-D8', 'D8-D9', 'D9-D10', 'D10-D11', 'D11-D12', 'G1-G2', 'G2-G3', 'G3-G4', 'G4-G5', 'G5-G6', 'G6-G7', 'G7-G8', 'G8-G9', 'G9-G10', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E5-E6', 'E6-E7', 'E7-E8', 'E8-E9', 'E9-E10', 'F1-F2', 'F2-F3', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F7-F8', 'F8-F9', 'F9-F10', 'F10-F11', 'F11-F12', 'H1-H2', 'H2-H3', 'H3-H4', 'H4-H5', 'H5-H6', 'H6-H7', 'H7-H8', 'H8-H9', 'H9-H10', 'J1-J2', 'J2-J3', 'J3-J4', 'J4-J5', 'J5-J6', 'J6-J7', 'J7-J8', 'J8-J9', 'J9-J10', 'J10-J11', 'J11-J12', 'O1-O2', 'O2-O3', 'O3-O4', 'O4-O5', 'O5-O6', 'O6-O7', 'O7-O8', 'O8-O9', 'O9-O10', 'O10-O11', 'O11-O12', 'I1-I2', 'I2-I3', 'I3-I4', 'I4-I5', 'I5-I6', 'I6-I7', 'I7-I8', 'I8-I9', 'I9-I10', 'I10-I11', 'I11-I12', 'I12-I13', 'I13-I14', 'I14-I15', 'I15-I16', 'I16-I17', 'I17-I18'],
"18":['F1-F2', 'F2-F3', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F7-F8', 'S1-S2', 'S2-S3', 'S3-S4', 'S4-S5', 'S5-S6', 'S6-S7', 'S7-S8', 'S8-S9', 'S9-S10', 'S10-S11', 'S11-S12', 'H1-H2', 'H2-H3', 'H3-H4', 'H4-H5', 'H5-H6', 'H6-H7', 'H7-H8', 'M1-M2', 'M2-M3', 'M3-M4', 'M4-M5', 'M5-M6', 'M6-M7', 'M7-M8', 'U1-U2', 'U2-U3', 'U3-U4', 'U4-U5', 'U5-U6', 'U6-U7', 'U7-U8', 'A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A6-A7', 'A7-A8', 'A8-A9', 'A9-A10', 'A10-A11', 'A11-A12', 'A12-A13', 'A13-A14', 'A14-A15', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'B9-B10', 'J1-J2', 'J2-J3', 'J3-J4', 'J4-J5', 'J5-J6', 'J6-J7', 'J7-J8', 'J8-J9', 'J9-J10', 'O1-O2', 'O2-O3', 'O3-O4', 'O4-O5', 'O5-O6', 'O6-O7', 'O7-O8', 'O8-O9', 'O9-O10', 'O10-O11', 'O11-O12', 'O12-O13', 'O13-O14', 'O14-O15', 'T1-T2', 'T2-T3', 'T3-T4', 'T4-T5', 'T5-T6', 'T6-T7', 'T7-T8', 'T8-T9', 'T9-T10', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-L6', 'L6-L7', 'L7-L8', 'L8-L9', 'L9-L10', 'P1-P2', 'P2-P3', 'P3-P4', 'P4-P5', 'P5-P6', 'P6-P7', 'P7-P8', 'P8-P9', 'P9-P10', 'P10-P11', 'P11-P12', 'P12-P13', 'P13-P14', 'P14-P15', 'Q1-Q2', 'Q2-Q3', 'Q3-Q4', 'Q4-Q5', 'Q5-Q6', 'Q6-Q7', 'Q7-Q8', 'Q8-Q9', 'Q9-Q10', 'Q10-Q11', 'Q11-Q12', 'Q12-Q13', 'Q13-Q14', 'Q14-Q15', 'X1-X2', 'X2-X3', 'X3-X4', 'X4-X5', 'X5-X6', 'X6-X7', 'X7-X8', 'X8-X9', 'X9-X10', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'Y1-Y2', 'Y2-Y3', 'Y3-Y4', 'Y4-Y5', 'Y5-Y6', 'Y6-Y7', 'Y7-Y8', 'Z1-Z2', 'Z2-Z3', 'Z3-Z4', 'Z4-Z5', 'Z5-Z6', 'Z6-Z7', 'Z7-Z8', 'Au1-Ref Au', 'Au2-Ref Au'],
"19":['A1-A2', 'A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A6-A7', 'A7-A8', 'B1-B2', 'B2-B3', 'B3-B4', 'B4-B5', 'B5-B6', 'B6-B7', 'B7-B8', 'B8-B9', 'B9-B10', 'B10-B11', 'B11-B12', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'C10-C11', 'C11-C12', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'D5-D6', 'D6-D7', 'D7-D8', 'D8-D9', 'D9-D10', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E5-E6', 'E6-E7', 'E7-E8', 'E8-E9', 'E9-E10', 'E10-E11', 'E11-E12', 'F1-F2', 'F2-F3', 'F3-F4', 'F4-F5'],
"20":['A1-A2', 'A2-A3', 'A6-A7', 'A7-A8', 'A8-A9', 'A10-A11', 'A11-A12', 'B1-B2', 'B2-B3', 'B3-B4', 'B7-B8', 'B8-B9', 'B9-B10', 'B10-B11', 'C1-C2', 'C2-C3', 'C3-C4', 'C4-C5', 'C5-C6', 'C6-C7', 'C7-C8', 'C8-C9', 'C9-C10', 'C10-C11', 'C11-C12', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'D5-D6', 'D6-D7', 'D7-D8', 'D8-D9', 'D9-D10', 'D10-D11', 'D11-D12', 'J1-J2', 'J2-J3', 'J3-J4', 'J8-J9', 'J9-J10', 'J10-J11', 'J11-J12', 'J12-J13', 'J13-J14', 'J14-J15', 'K1-K2', 'K2-K3', 'K3-K4', 'K4-K5', 'K5-K6', 'K6-K7', 'K7-K8', 'K8-K9', 'K9-K10', 'K10-K11', 'K11-K12', 'K12-K13', 'K13-K14', 'P1-P2', 'P2-P3', 'P3-P4', 'P4-P5', 'P5-P6', 'P6-P7', 'P7-P8', 'P8-P9', 'P9-P10', 'P10-P11', 'P11-P12', 'O1-O2', 'O2-O3', 'O3-O4', 'O4-O5', 'O5-O6', 'O6-O7', 'O7-O8', 'F3-F4', 'F4-F5', 'F5-F6', 'F6-F7', 'F9-F10', 'F10-F11', 'F11-F12', "A'1-A'2", "A'2-A'3", "A'3-A'4", "A'4-A'5", "A'8-A'9", "A'9-A'10", "A'10-A'11", "A'11-A'12", "B'1-B'2", "B'2-B'3", "B'3-B'4", "B'7-B'8", "B'8-B'9", "B'9-B'10", "C'1-C'2", "C'2-C'3", "C'3-C'4", "C'4-C'5", "C'5-C'6", "C'6-C'7", "C'7-C'8", "C'8-C'9", "C'9-C'10"],
"21":['X1-X2', 'X2-X3', 'X3-X4', 'X4-X5', 'X5-X6', 'X6-X7', 'X7-X8', 'X10-X11', 'X11-X12', 'X12-X13', 'X13-X14', 'X14-X15', 'X15-X16', 'X16-X17', 'X17-X18', 'I1-I2', 'I4-I5', 'I5-I6', 'I8-I9', 'I9-I10', 'A1-A2', 'A2-A3', 'A3-A4', 'A7-A8', 'A8-A9', 'A9-A10', 'A10-A11', 'T1-T2', 'T2-T3', 'T5-T6', 'T6-T7', 'T7-T8', 'T8-T9', 'E1-E2', 'E2-E3', 'E3-E4', 'E4-E5', 'E6-E7', 'E7-E8', 'E8-E9', 'B1-B2', 'B2-B3', 'B3-B4', 'B8-B9', 'B9-B10', 'B10-B11', 'C1-C2', 'C2-C3', 'C3-C4', 'C7-C8', 'C8-C9', 'C9-C10', 'C10-C11', 'D1-D2', 'D2-D3', 'D3-D4', 'D4-D5', 'D8-D9', 'D9-D10', 'D10-D11', 'S1-S2', 'S2-S3', 'S3-S4', 'S4-S5', 'S12-S13', 'S13-S14', 'S14-S15', 'S15-S16', 'S16-S17', 'Q1-Q2', 'Q2-Q3', 'Q3-Q4', 'Q9-Q10', 'Q10-Q11', 'Q11-Q12', 'Q12-Q13', 'Q13-Q14', 'O1-O2', 'O2-O3', 'O3-O4', 'O4-O5', 'O5-O6', 'O6-O7', 'O7-O8', 'O8-O9', 'O9-O10', 'O10-O11', 'O11-O12', 'P1-P2', 'P2-P3', 'P3-P4', 'P4-P5', 'P5-P6', 'P11-P12', 'P12-P13', 'P13-P14', 'P14-P15']
}

# original sampling from subsmeta.xlsx table
subject_fs = {
"1":500,
"2":500,
"3":512,
"4":500,
"5":500,
"6":500,
"7":500,
"8":500,
"9":500,
"10":500,
"11":2048,
"12":500,
"13":500,
"14":500,
"15":500,
"16":500,
"17":500,
"18":512,
"19":250,
"20":500,
"21":1024}

resampling = 512 if subject_fs[subject_id] in [512, 2048] else 500

connectivity_measures = ["PAC", "PEC"] if bands is None else ["SCR", "SCI", "PLV", "PLI", "CC"]  

for measure in connectivity_measures:

    timer_start()
    print(measure)
    
    prep_data = REc.load(file_path).data
    epochs = prep_data.x_prep

    cm = struct(y=prep_data.y, i=prep_data.i, nodes=prep_data.nodes)

    if measure == "SCR":
        cm._set(X = connectivity_analysis(epochs, spectral_coherence, fs=resampling, imag=False))

    elif measure == "SCI":
        cm._set(X = connectivity_analysis(epochs, spectral_coherence, fs=resampling, imag=True))

    elif measure == "PEC": 
        parallelize = Parallel(n_jobs=-1)(delayed(PEC)(ep,i+1) for i,ep in enumerate(epochs))
        cm_pec = [p for p in parallelize]
        cm._set(X = cm_pec)

    elif measure in "PLV":
        cm._set(X = connectivity_analysis(epochs, phaselock))

    elif measure == "PLI":
        cm._set(X = connectivity_analysis(epochs, phaselag))

    elif measure == "CC":
        cm._set(X = connectivity_analysis(epochs, cross_correlation))

    elif measure == "PAC":
        cm._set(X = connectivity_analysis(epochs, PAC, fs=resampling))

    reduced_sz_cms, reduced_base_cms = [],[]

    if prep_data.extra_nodes_1:

        extra_nodes_sz = sorted([nodes[subject_id].index(label) for label in prep_data.extra_nodes_1], reverse=True) if isinstance(prep_data.extra_nodes_1[0], str) else sorted(prep_data.extra_nodes_1, reverse=True)

        for m in extra_nodes_sz:
            if isinstance(m, str): reduced_sz_cms = exclude_node_from_cm(cm.X[0:int(len(cm.X)/2)], nodes[subject_id].index(m))
            elif isinstance(m, int): reduced_sz_cms = exclude_node_from_cm(cm.X[0:int(len(cm.X)/2)], m)

        cm.X = reduced_sz_cms+cm.X[int(len(cm.X)/2)::]

    if prep_data.extra_nodes_0:

        extra_nodes_base = sorted([nodes[subject_id].index(label) for label in prep_data.extra_nodes_0], reverse=True) if isinstance(prep_data.extra_nodes_0[0], str) else sorted(prep_data.extra_nodes_0, reverse=True)

        for m in extra_nodes_base:
            if isinstance(m, str): reduced_base_cms = exclude_node_from_cm(cm.X[int(len(cm.X)/2)::], nodes[subject_id].index(m))
            elif isinstance(m, int): reduced_base_cms = exclude_node_from_cm(cm.X[int(len(cm.X)/2)::], m)

        cm.X = cm.X[0:int(len(cm.X)/2)]+reduced_base_cms
    
    print(f"Seizure epoch (shape {cm.X[0].shape}): {cm.X[0]}")
    print(f"Baseline epoch (shape {cm.X[-1].shape}): {cm.X[-1]}")
    
    path_cm = main_folder + "/connectivity_matrices/"
    makedirs(main_folder + "/connectivity_matrices/", exist_ok=True)

    if bands is not None: REc(cm).save(path_cm + f"{subject_id}-{woi_code[woi]}-{measure}-{bands}.prep") 
    elif bands is None: REc(cm).save(path_cm + f"{subject_id}-{woi_code[woi]}-{measure}.prep")

    timer_end()