from inspect import ismethod
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

this = Record

from numpy import array, average, median, random
import scipy.stats as stats

class Table(COb, GEq, GRe):
    data = None
    default = None
    PAD = 3
    ELLIPSIS_AT = int(GRe._lines_*.3)
    class _axes(list):
        def insert(_from, this, item):
            super().insert(this, item)
            _from.__dict__[item.name] = item
        def __setitem__(_, pos, axis):
            super().__setitem__(pos, axis)
            _.__dict__[axis.name] = axis
    class axis(list, GSet, GRe):
        name = None
        root = None
        _to = 0
        def __init__(axis, root, labels, name='ax', force_at=None):
            if force_at: root.axes[force_at].name = None
            with this(labels) as dim:
                if not dim.isiterable and dim.inherits(int): labels = range(labels)
            super().__init__(labels)
            names = [ax.name for ax in root.axes]
            name_, n = name, 1
            while name in names: name = name_ + str(n); n+=1
            axis._set(root=root, name=name)
            if force_at: root.axes[force_at] = axis
            else: root.axes.insert(0, axis)
        def at(axis, field):
            field = int(field) if this(field).inherits(str) and field.isdecimal() else field
            found = axis.index(field) if field in axis else None
            axis._to = found if found is not None else field
        def __repr__(_):
            return '{}: {}'.format(_.name, _._resize(' '.join([str(i) for i in _])))
    def __init__(this, **table_description):
        super().__init__(axes=this._axes())
        this.set(**table_description)
    def reset(data):
        base = None
        if len(data.axes)>0:
            base = [data.default]*len(data.axes[-1])
            for ax in reversed(data.axes[0:-1]): base = [base]*len(ax)
        data._set(data=array(base))
    @property
    def ax_names(_): return [ax.name for ax in _.axes]
    def at(data, axis):
        with this(axis) as _axis:
            if _axis.inherits(int):
                if axis>0 and axis<len(data.axes): return data.axes[axis]
            elif _axis.inherits(str):
                axes = data.ax_names
                if axis in axes: return data.axes[axes.index(axis)]
        return None
    def _check(build):
        if build.data is None: build.reset()
        return build.data
    def _find(_, inverted, ax_field):
        _._check()
        def index(axis, entry):
            fields = _.at(axis)
            if fields is not None:
                if this(entry).isiterable:
                    return tuple([fields.index(field) for field in entry])
                else: return ':'
            return None
        def translate(axis, found):
            if axis.name in found:
                _range = found[axis.name]
                if this(_range).inherits(tuple):
                    if inverted: found[axis.name] = tuple([field for field in range(len(axis)) if field not in _range])
                    return "_from['{}']".format(axis.name)
            return ':'
        found={field:index(field,entry) for field,entry in ax_field.items()}
        found={field:value for field,value in found.items() if value is not None}
        reshape='M['+','.join([translate(axis,found) for axis in _.axes])+']'
        _._set(_reshape_ = (reshape, found))
    def _by_val(_, depth=-1, _layer=0):
        M, axes = _._check(), _.axes
        do, _from = _.__dict__.pop('_reshape_') if '_reshape_' in _._meta() else (None, {})
        copy = super()._by_val(depth, _layer)
        copy.axes = []
        for ax in reversed(axes):
            fields = [field for n,field in enumerate(ax) if n in _from[ax.name]] if ax.name in _from else ax
            copy.set(**{ax.name:fields})
        copy.data = eval(do) if do else M.copy()
        return copy
    def _translate(_, directions):
        axes = directions.split(',')
        for ax_dir in axes:
            ax, field = [token.strip() for token in ax_dir.split(':')]
            axis = _.at(ax)
            if axis: axis.at(field)
        return '['+','.join([str(ax._to) for ax in _.axes])+']'
    def _get_set(_, directions, mode='get', value=None):
        if mode == 'get' and not '_MGET' in _.sets: _._MGET = []
        if len(directions) == len(_.axes):
            resolve, message = True, []
            for n,part in enumerate(directions):
                _part = this(part)
                if _part.inherits(int, str) or _part.isiterable and len(part)==1:
                    token = part if _part.inherits(str, int) else part[0]
                    message.append(':'.join([str(_.axes[n].name),str(part)]))
                else:
                    resolve = False
                    for token in part:
                        redirection = list(directions)
                        redirection[n] = token
                        _._get_set(tuple(redirection), mode, value)
            if resolve:
                message = ','.join(message)
                if mode=='get': _._MGET.append(_[message])
                else: _[message] = value
    def __getitem__(by, field_directions):
        M = by._check()
        if this(field_directions).inherits(tuple):
            by._get_set(field_directions)
            result = by.__dict__.pop('_MGET')
            return result
        else: return eval('M'+by._translate(field_directions))
    def __setitem__(by, field_directions, value):
        M = by._check()
        if this(field_directions).inherits(tuple): by._get_set(field_directions, 'set', value)
        else: exec('M'+by._translate(field_directions)+'=value')
    def set(data, **ax_field):
        for name, fields in ax_field.items(): data.axis(data, fields, name)
    def get(data, **ax_field):
        data._find(0, ax_field)
        return data._by_val()
    def let(data, **ax_field):
        data._find(1, ax_field)
        return data._by_val()
    @property
    def sets(tree): return set(meta(tree))
    def __repr__(self):
        M = self._check()
        _repr, dimensions = '', len(self.axes)
        if not dimensions: _repr += 'void table\n'
        else:
            dimensions = len(self.axes)
            y = self.axes[-2] if dimensions >= 2 else None
            if dimensions>2:
                y = self.axes[-2]
                for n,ax in enumerate(self.axes[:-2]): _repr += '{}{}: {}/{}\n'.format('\t'*n, ax.name, ax._to, len(ax))
            mr = eval('M'+str([ax.index(ax._to) for ax in self.axes][:-2])) if dimensions>2 else M
            pad = max([len(y.name)]+[len(str(field)) for field in y]+[len(str(value)) for line in mr for value in line])+self.PAD if dimensions>1 else 0
            _repr, x, spaces = _repr+y.name+'\n' if y else '', self.axes[-1], ' '*pad if pad>0 else '\t'
            header = spaces+''.join([str(field).ljust(pad) for field in x])
            _repr += self._resize(header) + '\n'
            ellipsis_at = self._lines_-self.ELLIPSIS_AT-1
            last_values_from = len(mr)-self.ELLIPSIS_AT
            if last_values_from<=ellipsis_at: last_values_from = ellipsis_at+1
            for n, line in enumerate(mr):
                if n<ellipsis_at or n>last_values_from:
                    values = str(y[n]).ljust(pad) if y else ''
                    values += ''.join([str(value).ljust(pad) for value in line])
                    _repr += self._resize(values) + '\n'
                elif n==ellipsis_at:
                    _repr += self._ellipsis_ + '\n'
        extra = {k:v for k,v in meta(self).items() if k != 'data' and k != 'axes'}
        _repr += self._resize(spaces*len(x)+x.name)+'\n'+'\n'.join(['{}:\t{}'.format(k, self._repr(v)) for k,v in extra.items()])
        return _repr
TAb = tab = Table

def set_repr_to(lines, ratio=.7):
    set_repr_to(lines)
    Table.ELLIPSIS_AT = int(Table._lines_*(1-ratio))

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


import pyedflib as edf

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
            eeg._set(data=array(data), at_epoch=(n, epoch()))
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
            if p<=0.05: return p, median(eeg.deltas)
            return p, average(eeg.deltas)    
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
            if 'seed' in sets: random.seed(_.seed)
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
                at = map.pool[event].pop(random.randint(len(map.pool[event])))
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

def preprocess(eeg, epoch, limit=500): 
    """Primary preprocessing. Resamples data to a limit frequency and applies a notch filter.

    Args:
        eeg (eeg): Wrapper object of the raw EEG data and metadata.
        epoch (list): Signal epoch.
        limit (int): Target frequency for resampling. Defaults to 500.

    Returns:
        list: Preprocessed epoch.
    """
    sampling, rse = limit, epoch
    if eeg.fs == limit: rse = epoch
    elif eeg.fs%limit != 0: rse = upsample(epoch, eeg.fs, limit) if eeg.fs<limit else dwindle(epoch, int(eeg.fs/limit)-1) 
    else: rse = upsample(epoch, eeg.fs, limit) if eeg.fs<limit else dwindle(epoch, int(eeg.fs/limit)-2) 
    nse = notch(rse, fs=sampling, order=2)
    return nse

from numpy import correlate, average, array, angle, mean, sign, exp, zeros, abs, unwrap, fromfile, unpackbits, packbits
from scipy.signal import hilbert, csd

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

# --------------------------------------------------------------------------------------------------------------------------------------------------

from sys import argv
from os import getcwd, makedirs

woi = argv[3]

main_folder = getcwd()

file_seizure = main_folder + f"/data/seizure/{argv[1]}-seizure.EDF"
file_baseline = main_folder + f"/data/baseline/{argv[2]}-baseline.EDF"

subject_id = file_seizure.split("/")[-1].split("-")[0]

span, step = 1000, 500      # in ms
min_woi_duration = 60000    # in ms
n_epochs = int(((min_woi_duration/step)-1) / 2)

eeg_seizure = EEG.from_file(file_seizure, epoch(ms(step), ms(span)))    # load raw seizure SEEG data as an EEG object (class) 
eeg_baseline = EEG.from_file(file_baseline, epoch(ms(step), ms(span)))   # load raw baseline SEEG data as an EEG object (class) 

nodes_seizure = list(eeg_seizure.axes.region)
nodes_baseline = list(eeg_baseline.axes.region)
montage_overlap = list(set(nodes_seizure) & set(nodes_baseline))
extra_in_baseline = [ch for ch in nodes_baseline if ch not in montage_overlap]
extra_in_seizure = [ch for ch in nodes_seizure if ch not in montage_overlap]

if extra_in_baseline: 
    for chn in extra_in_baseline: eeg_baseline.axes.region.remove(chn)

if extra_in_seizure: 
    for chn in extra_in_seizure: eeg_seizure.axes.region.remove(chn)

nodes = montage_overlap

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

eeg_seizure._set(fs = subject_fs[subject_id])
eeg_baseline._set(fs = subject_fs[subject_id])
fs_min = min(eeg_seizure.fs, eeg_baseline.fs)
# set the desired resampled frequency to 500 Hz if sampling is not already 512 Hz
resampling = 512 if fs_min==512 else 500

notes_seizure = [note for note in eeg_seizure.notes]
notes_baseline = [note for note in eeg_baseline.notes]
sz_start_note, sz_end_note, base_center_note, base_end_note = 'EEG inicio', 'EEG fin', 'mitad-NS', 'NS-fin'

if sz_start_note not in notes_seizure:
    altnote = [a for a in notes_seizure if sz_start_note in a]
    sz_start_note = altnote[0]
if sz_end_note not in notes_seizure:
    altnote = [a for a in notes_seizure if sz_end_note in a]
    sz_end_note = altnote[0]
if base_center_note not in notes_baseline:
    altnote = [a for a in notes_baseline if base_center_note in a]
if base_end_note not in notes_baseline:
    altnote = [a for a in notes_baseline if base_end_note in a]

SET(eeg_seizure, _as='N')                       # N - baseline (non-seizure)
SET(eeg_seizure, sz_start_note, 'W')            # W - WOI
SET(eeg_seizure, sz_end_note, 'S', epoch.END)   # S - seizure

SET(eeg_baseline, _as='N')
SET(eeg_baseline, base_center_note, 'W')         # W - middle point
SET(eeg_baseline, base_end_note, 'S', epoch.END) # S - terminal point (end of recording)

eeg_seizure.optimize()
eeg_seizure.remap()

eeg_baseline.optimize()
eeg_baseline.remap()

units = int((eeg_seizure.notes[sz_end_note][0].time - eeg_seizure.notes[sz_start_note][0].time)*(span/step))

woi_code = {'1':"baseline", '2':"preseizure5", '3':"preseizure4", '4':"preseizure3", '5':"preseizure2", '6':"preseizure1", '7':"transition1", '8':"transition2", '9':"transition60", '10':"seizure"}
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

a, ai = eeg_seizure.sample.get('W', n_epochs)   # fetch epochs and epoch indices
b, bi = eeg_baseline.sample.get('W', n_epochs)  
i = ai + bi                                     # save indices                  
x = a + b                                       # save epochs
y = [1]*n_epochs + [0]*n_epochs                 # save epoch labels (1 for seizure, 0 for baseline)

pp_seizure = [preprocess(eeg_seizure, ep, resampling) for i,ep in enumerate(a)] 
pp_baseline = [preprocess(eeg_baseline, ep, resampling) for i,ep in enumerate(b)] 

connectivity_measures_filtered = ["SCR", "SCI" "PLV", "PLI", "CC"]  

for measure in connectivity_measures_filtered:

    for bands in [(0,4),(4,8),(8,13),(13,30),(30,70),(70,150)]:

        fpp_seizure = [band(e, bands, pp_seizure[0].shape[1]) for e in pp_seizure]
        fpp_baseline = [band(e, bands, pp_baseline[0].shape[1]) for e in pp_baseline]

        cm = struct(x=array(x), y=array(y), i=array(i)) # initiating an object for storing a connecivity matrix with shape (x, y) and epoch indices

        cm._set(nodes = nodes)

        epochs = fpp_seizure + fpp_baseline

        if measure == "SCR":
            cm._set(X = connectivity_analysis(epochs, spectral_coherence, fs=resampling, imag=False))

        elif measure == "SCI":
            cm._set(X = connectivity_analysis(epochs, spectral_coherence, fs=resampling, imag=True))

        elif measure == "PEC": 
            parallelize = Parallel(n_jobs=-1)(delayed(PEC)(ep,i+1) for i,ep in enumerate(fpp_seizure))
            cm_pec = [p for p in parallelize]
            cm._set(X = cm_pec)

        elif measure in "PLV":
            cm._set(X = connectivity_analysis(epochs, phaselock))

        elif measure == "PLI":
            cm._set(X = connectivity_analysis(epochs, phaselag))

        elif measure == "CC":
            cm._set(X = connectivity_analysis(epochs, cross_correlation))

        elif measure == "PAC":
            cm._set(X = connectivity_analysis(epochs, PAC, dtail=True, fs=resampling))

        reduced_sz_cms, reduced_base_cms = [],[]

        for m in extra_in_seizure: reduced_sz_cms = exclude_node_from_cm(cm.X[0:int(len(cm.X)/2)], nodes_seizure.index(m))
        for m in extra_in_baseline: reduced_base_cms = exclude_node_from_cm(cm.X[int(len(cm.X)/2)::], nodes_baseline.index(m))

        cm.X = reduced_sz_cms+reduced_base_cms

        path_cm = main_folder + "connectivity_matrices/"
        makedirs(path_cm, exist_ok=True)
        REc(cm).save(path_cm + f"{subject_id}-{woi_code[woi]}-{measure}-{bands}.prep".replace(" ","")) 


connectivity_measures_unfiltered = ["PAC"] 

for measure in connectivity_measures_unfiltered:
    cm = struct(x=array(x), y=array(y), i=array(i)) # initiating an object for storing a connecivity matrix with shape (x, y) and epoch indices

    cm._set(nodes = nodes)

    epochs = pp_seizure + pp_baseline

    if measure == "PEC": 
        parallelize = Parallel(n_jobs=-1)(delayed(PEC)(ep,i+1) for i,ep in enumerate(fpp_seizure))
        cm_pec = [p for p in parallelize]
        cm._set(X = cm_pec)

    elif measure == "PAC":
        cm._set(X = connectivity_analysis(epochs, PAC, dtail=True, fs=resampling))

    reduced_sz_cms, reduced_base_cms = [],[]

    for m in extra_in_seizure: reduced_sz_cms = exclude_node_from_cm(cm.X[0:int(len(cm.X)/2)], nodes_seizure.index(m))
    for m in extra_in_baseline: reduced_base_cms = exclude_node_from_cm(cm.X[int(len(cm.X)/2)::], nodes_baseline.index(m))

    cm.X = reduced_sz_cms+reduced_base_cms  
    path_cm = main_folder + "connectivity_matrices/"
    makedirs(path_cm, exist_ok=True)
    REc(cm).save(path_cm + f"{subject_id}-{woi_code[woi]}-{measure}.prep") 