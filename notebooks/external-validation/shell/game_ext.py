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
import matplotlib.pyplot as plt
from random import randint, sample
from scipy.interpolate import interp1d
from pickle import dump, load

class Player:
    def __init__(self, AI, deck, name, n_in_hand=5):
        """
        Initializes a Player object.

        Parameters:
        - AI: The artificial intelligence strategy used by the player.
        - deck: The deck of cards available for the player.
        - name: The name of the player.
        - n_in_hand: The number of cards initially dealt to the player's hand.
        """
        self.deck = deck
        self.n_in_hand = n_in_hand
        self.logic = AI
        self.cards = []  # Player's hand
        self.score = []  # Player's score (not used in the provided code)
        self.name = name
        
    def shuffle(hand):
            hand.cards = sorted(hand.deck)

    def check(scores):
        """
        Parameters:
        - scores: The scores to be checked.

        Returns:
        - bool: True if the scores are real, otherwise False.
        """
        return scores.real

    def play(god, *players):
        """
        Determines the player's move in a game.

        Parameters:
        - god: The player making the move.
        - players: Other players in the game.
 
        Returns:
        - int: The chosen card to play.
        """
        best, other = max(god.cards), []
        for player in players:
            other += player.cards

        if best >= max(other):
            return best
        else:
            return min(god.cards)

    def card(draw):
        """
        Draws a card based on the player's AI logic.

        Parameters:
        - draw: The drawing logic.

        Returns:
        - int: The drawn card.
        """
        return draw.logic(draw)

def rn_choice(logic):
    return logic.cards.pop(randint(0,len(logic.cards)-1))

def deck_average(of): return sum(of)/len(of)

def deck_mm(of): return min(of)*max(of)/np.mean(of)

def deck_coef_of_variation(of): return np.std(of)/np.mean(of)

def deck_max(of): return np.max(of)

def deck_random(of): return of[randint(0,len(of)-1)]

def pop_closest_card(cards, summary_stat):
    if not cards: return None
    # The min function then compares the tuples for each card, and it selects the card with the minimum tuple value. 
    # The first element of the tuple (abs(card - summary_stat)) is used as the primary key, and in case of a tie, the second element (-card) is used as a tiebreaker.
    closest_value = min(cards, key=lambda cards: (abs(cards - summary_stat), -cards))
    cards.remove(closest_value)
    return closest_value

def play(*game):
    # Initialize each player's real and best score attributes to 0
    for player in game:
        player.real, player.best = 0, 0
        player.shuffle()

    # Define a function 'resolve' to find the winners among the players
    def resolve(hand, by=[], best=-1):
        for player, card in hand:
            # Check if the card is equal to the best card, add player to 'by' list
            if best == card:
                by.append(player)
            # If the card is greater than the best, update 'best' and set 'by' to [player]
            if best < card:
                best = card
                by = [player]
        return by

    # Use the 'resolve' function to find the winners among the players
    winners = resolve([(player, player.card()) for player in game])

    # Increment the 'real' attribute for each winner
    for player in winners:
        player.real += 1

    # Update each player's score by appending the result of the 'check' method
    for player in game:
        player.score.append(player.check())

# -----------------------------------------------------------------------------------

from os import makedirs, getcwd
from sys import argv

# Set a specific random seed (e.g., 42 for reproducibility)
random_seed = 42
np.random.seed(random_seed)

main_folder = getcwd() + "/"

path_game_scores = main_folder + "game_scores/"

makedirs(path_game_scores, exist_ok=True)

sub = argv[1]

print(f"\nSubject: {sub}")

df = pd.read_csv(main_folder + f"cvs_pairs_ext.csv")
df_sub = df[df['Subject'] == int(sub)]
connectivity_measures = list(df['CM'].unique())

players = list(set(df_sub.Pair))

rounds, turns, n_cards = 100, 10, 24
strategy_ext = ["mm"]
summary_stat_label = ["mm"]

all_nodes = REc.load(main_folder + f"result/{sub}-SCR-(1,4).res").data.data.nodes
r = 0.1
group_size = int(len(all_nodes)*r)
# print("Random group size =", group_size)

# Define players as random groups of nodes
players = [sorted(list(np.random.choice(all_nodes, size=group_size, replace=False))) for i in range(len(all_nodes)*5)] 

for i,summary_stat in enumerate([deck_mm]):

    for measure in connectivity_measures:
        print(f"Subject: {sub}")
        print(f"Connectivity measure: {measure}")
        print(f"Summary statistic for connectivity change: {summary_stat_label[i]}")

        df_sub_cm = df_sub[df_sub['CM'] == measure]

        hands = [] # initialize hands, a list of tuples (player, deck)

        for group_labels in players:
            if group_labels not in [hand[0] for hand in hands]:

                # Split the Labels column into two nodes using '<->'
                split_labels = df_sub_cm['Labels'].str.split('<->', expand=True)

                # Check if both nodes are in group_labels
                relevant_rows = df_sub_cm[((split_labels[0].isin(group_labels)) & split_labels[1].isin(group_labels))]
                # Calculate (max(CVS)+min(CVS)) / mean(CVS) scores (MM scores)
                group_deck = list(relevant_rows['CVS'].apply(lambda x: summary_stat(list(map(float, x.strip('[]').split())))))
                hands.append((group_labels, group_deck))

        n_cards = len(hands[0][1]) # number of cards in each player's deck

        remap = interp1d([0,1],[1,100]) # remap values from 0 to 1, to 1 to 100
        
        game_score = {tuple(p):[] for p in [hand[0] for hand in hands]}

        rounds=10
        for turn in range (rounds):

            for r in range(n_cards):

                game = [Player(AI=rn_choice, deck=hand[1], name=tuple(hand[0]), n_in_hand=n_cards) for hand in hands]

                play(*game)

                scores = sorted([(player.name, player.score) for player in game], key=lambda x:x[1], reverse=True)

                top_score, fall = scores[0][1], 0
                for name, score in scores:
                    if score==top_score: game_score[name].append(1); fall+=1
                    elif score!=top_score: break
                for name,score in scores[fall:]: game_score[name].append(0)

        sorted_players = {k:v for k, v in sorted(game_score.items(), key=lambda item: sum(item[1]), reverse=True)}
        dump(sorted_players, open(path_game_scores + f"scores_{measure}_{summary_stat_label[i]}_{n_cards}cards_{rounds}rounds_sub{sub}.p", "wb"))
