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

import pandas as pd
import numpy as np
from sys import argv
from os import getcwd
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from collections import Counter

from glob import glob
import re
from pickle import load

import pandas as pd


def get_non_operated(main_folder):
    try:
        # Read the Excel file
        df = pd.read_excel(main_folder + 'metadata.xlsx')
        # Filter rows where OUTCOME is -1
        outcome_minus_one = df[df['OUTCOME'] == -1]
        # Get the SCC_ID values from the filtered rows
        nonop = outcome_minus_one['SCC_ID'].tolist()
        print(nonop)
        return nonop
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_outcome_for_subject(sub, main_folder):
    try:
        df = pd.read_excel(main_folder + 'metadata.xlsx')
        # Find the row corresponding to the subject ID
        row = df[df['SCC_ID'] == sub]
        # Get the outcome from the row
        outcome = row['OUTCOME'].values[0] if not row.empty else None
        return outcome
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_soz_labels(sub, main_folder):
    try:
        df = pd.read_excel(main_folder + "metadata.xlsx")
        
        # Find the row corresponding to the given sub
        row = df[df['SCC_ID'] == sub]
        # Extract the SOZ labels from the row
        soz_labels = row['SOZ'].values[0]
        # Split the string by comma and remove spaces
        soz_labels = soz_labels.replace(' ', '').split(',')
        # Convert all letters to uppercase
        soz_labels = [label.upper() for label in soz_labels]
        return soz_labels
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def check_winners(sorted_players, sigmas, resection, group_size):
    """
    Check winners' performance based on cross-validation scores and resection overlap.

    Parameters:
    - sorted_players (dict): Dictionary containing sorted players' cross-validation scores.
    - sigmas (float): Number of standard deviations for defining the threshold.
    - nodes (list): List of node labels.
    - resection (list): List of nodes in the resection area.
    - group_size (float): Player or random group node number.

    Returns:
    - Tuple: (number of winners above threshold, winner-loser ratio, median resection overlap, mean resection overlap).
    """

    # Calculate group scores and resection matches
    group_scores = {player:sum(sorted_players[player]) for player in sorted_players}
    resection_match = {player: [n for n in player if n in resection] for player in group_scores}

    # Calculate statistical measures
    median = np.median(list(group_scores.values()))
    mean = np.average(list(group_scores.values()))
    std = np.std(list(group_scores.values()))

    # Define threshold based on mean and standard deviations
    thresh = mean + (std * sigmas)
    winners_above_thresh = [players for players in group_scores if group_scores[players] >= thresh]
    N_winners_above_thresh = len(winners_above_thresh)
    print(f"Number of winners above threshold ({thresh}) = {N_winners_above_thresh}")

    if N_winners_above_thresh > 0:
        # Calculate resection overlap for winners and losers
        resection_overlap_winners = [len(resection_match[players]) / group_size for players in winners_above_thresh]
        resection_overlap_losers = [len(resection_match[players]) / group_size for players in resection_match if
                                    players not in winners_above_thresh]

        # Calculate the winner-loser ratio
        winner_loser_ratio = np.mean(resection_overlap_winners) / np.mean(resection_overlap_losers)
        print("Winner Resection Overlap/Loser Resection Overlap =", winner_loser_ratio)

        return N_winners_above_thresh, winner_loser_ratio, np.median(resection_overlap_winners), np.mean(resection_overlap_winners),

    else:
        return N_winners_above_thresh, 0, 0, 0


def to_labels(pos_probs, threshold):
    # function to map all values >=threshold to 1 and all values <threshold to 0

	return list((pos_probs >= threshold).astype('int')) 


def moving_thresh_auc(predictive_measure=[], outcome=[], n_good=14, n_bad=7, moving_step=0.00001):
    # returns AUC, best threshold, true negatives and true positives at the best threshold

    thresholds = np.arange(0, np.max(predictive_measure), moving_step)

    g = np.array([pm for i,pm in enumerate(predictive_measure) if outcome[i]=="good"])
    b = np.array([pm for i,pm in enumerate(predictive_measure) if outcome[i]=="bad"])

    A, A_top = 0, 0
    T = 0
    tp_top, tn_top = 0, 0
    step = 0
    for t in thresholds:    
        g_l, b_l = to_labels(g, t), to_labels(b, t)
        tp = sum(g_l)/n_good 
        tn = b_l.count(0)/n_bad
        A = (tp + tn)/2
        if A>A_top: 
            step=0
            A_top=A
            T=t
            tn_top,tp_top=tn,tp
        elif A==A_top: step+=moving_step

    return (A_top, T, tn_top, tp_top)
# ______________________________________________________________________________________________________

main_folder = getcwd() + "/"

sub = argv[1]
sigma = int(argv[2])

skip_subjects = [str(id) for id in get_non_operated(main_folder)]

summary_stat_label = ["mm"]

Subject, Outcome, CM, R, Group_size, Strategy, Sigmas, N_winners, Mean_overlap_ratio, Median_overlap_winners, Mean_overlap_winners = [],[],[],[],[],[],[],[],[],[],[]

if sub not in skip_subjects:

    path_game_scores = main_folder + "game_scores/"

    scores_files = glob(path_game_scores + f"*sub{sub}.p")

    print("Non-operated subjects skipped:", skip_subjects)

    for filename in scores_files:

        # "scores_{measure}_{summary_stat_label[i]}_{n_cards}cards_{rounds}rounds_preseizure1_sub{sub}.p"
        params = filename.split("/")[-1].split(".")[0].split("_")

        measure = params[1]
        summary_stat_label = params[2]
        n_cards = re.sub(r'[^0-9]', '', params[3])
        rounds = re.sub(r'[^0-9]', '', params[4])
        sub = re.sub(r'[^0-9]', '', params[5])

        surgical_outcome = get_outcome_for_subject(int(sub), main_folder)

        # we need resection_nodes and group_size to check the winners
        all_nodes = REc.load(main_folder + f"result/{sub}-PAC.res").data.data.nodes

        resection_nodes = get_soz_labels(int(sub), main_folder)
        print(resection_nodes)
        r = 0.1
        group_size = int(len(all_nodes)*r)

        # load sorted_players
        sorted_players = load(open(filename, "rb"))

        n_winners, winner_loser_ratio, resection_overlap_winners_median, resection_overlap_winners_mean = check_winners(sorted_players, sigma, resection_nodes, group_size)

        Subject.append(sub)
        Outcome.append(surgical_outcome)
        CM.append(measure)
        Strategy.append(summary_stat_label)
        Sigmas.append(sigma)

        N_winners.append(n_winners)
        Mean_overlap_ratio.append(winner_loser_ratio)
        Median_overlap_winners.append(resection_overlap_winners_median)
        Mean_overlap_winners.append(resection_overlap_winners_mean)

        surgical_outcome_prediciton_data = pd.DataFrame({"Subject":Subject, "Outcome":Outcome, "CM":CM, "Strategy":Strategy, "Sigmas":Sigmas, "N_winners":N_winners, "Mean_overlap_ratio":Mean_overlap_ratio, "Median_overlap_winners":Median_overlap_winners, "Mean_overlap_winners":Mean_overlap_winners})
        surgical_outcome_prediciton_data.to_excel(main_folder + f"surgical_outcome_data_{sigma}sigma_sub{sub}.xlsx")
