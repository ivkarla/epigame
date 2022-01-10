# pylint: disable=no-self-argument, no-member, unused-variable

from core import GhostSet, GSet, ClonableObject, COb, ComparableGhost, GEq, ReprGhost, GRe, struct
from core import copy_by_val, copy, by_val, meta, isfx
from core import set_repr_to as _set_repr_to
from core import Record as rec
this = rec

import numpy as np
import scipy.stats as stats

_DEBUG = ''

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
        data._set(data=np.array(base))
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
    _set_repr_to(lines)
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

if _DEBUG=='repr':
    t = tab(x=('a','b','c'), y=3)
    r = repr(t)
elif _DEBUG=='copy':
    t = tab(x=('a','b','c'), y=3)
    s = t._by_val()
    g = t.get(x=('a',))
    l = t.let(x=('a',))
elif _DEBUG=='get_set':
    t = tab(x=('a','b','c'), y=3)
    t['y:1,x:b'] = 'test'
    get = t['y:1,x:b']
elif _DEBUG=='mult_get_set':
    t = tab(x=('a','b','c'), y=3)
    t[1, t.axes.x] = 'test'
    get = t[t.axes.y, 0]