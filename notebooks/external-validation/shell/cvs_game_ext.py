#!/opt/venv/bin/python

from random import randint, sample

def av(of): return sum(of)/len(of)
def mm(of): return min(of)*max(of)/av(of)

class Player:
    def __init__(my, AI, deck, name, n_in_hand=5):
        my.deck, my.n_in_hand, my.logic, my.cards, my.score, my.name = deck, n_in_hand, AI, [], [], name
    def shuffle(hand):
        lo, hi = min(hand.deck), max(hand.deck)
        hand.cards = sorted(sample(hand.deck, hand.n_in_hand)) # draw a number of random cards from the deck
    def check(scores): return scores.real
    def play(god, *among):
        best, other = max(god.cards), []
        for player in among: other+=player.cards
        if best>=max(other): return best
        else:                return min(god.cards)
    def card(draw): return draw.logic(draw)

def rn_choice(logic):
    return logic.cards.pop(randint(0,len(logic.cards)-1))

def av_choice(logic):
    lo, hi = min(logic.deck), max(logic.deck)
    set, card, thresh = logic.cards, 0, (lo-1)+(1+hi-lo)/2
    actual_hand = av(logic.cards)
    if actual_hand>=thresh: card = set.pop(set.index(max(set)))
    else:                   card = set.pop(set.index(min(set)))
    return card

def mm_choice(logic):
    card, set = 0, logic.cards
    thresh, hi,lo = mm(set), max(set), min(set)
    if abs(hi-thresh)>abs(lo-thresh): card = set.pop(set.index(max(set)))
    else:                             card = set.pop(set.index(min(set)))
    return card

def mx_choice(logic):
    set = logic.cards
    return set.pop(set.index(max(set)))

def play(*game):
    for player in game: player.real,player.best=0,0; player.shuffle()
    def resolve(hand, by=[], best=-1):
        for player,card in hand:
            if best==card: by.append(player)
            if best <card: best=card; by=[player]
        return by
    winners = resolve([(player, player.card()) for player in game])
    for player in winners: player.real += 1
    for player in game: player.score.append(player.check())

# -------------------------------------------------------------------------------

import pandas as pd
from sys import argv
from os import getcwd, makedirs
from pickle import dump
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

main_folder = getcwd()
path_res = main_folder + "/result/"
path_deck = main_folder + "/decks/"
path_scores = main_folder + "/game_scores/"

makedirs(path_deck, exist_ok=True)
makedirs(path_scores, exist_ok=True)

sub = argv[1]
print(f"Subject ID: {sub}")

df = pd.read_csv(main_folder + "cvs_pairs_preseizure1.csv")

table_1 = df.groupby("Subject").get_group(sub) # table is a subject
conn_measures = list(set(table_1.CM))
players = list(set(table_1.Pair))

def create_deck(node_pair, connectivity):

    cvs = table_1.groupby("Pair").get_group(node_pair).groupby("CM").get_group(connectivity).reset_index().CVS[0]

    ones = cvs.count("1.")
    cvs = cvs.replace("1.","")

    a = [x for y in [c.split("0") for c in cvs[1:-1].replace(" ","").split("\n")] for x in y]
    cards=[1.0 for i in range(ones)]
    for s in a:
        if s=="": pass
        elif s[-2::]=="1.": cards.append(float("0"+s[:-2])); cards.append(float(s[-2::]))
        else: cards.append(float("0"+s))

    return cards

game = []
player_deck = {pair:[] for pair in players}

for node_pair in players:

    parallelize = Parallel(n_jobs=-1)(delayed(create_deck)(node_pair,connectivity)for connectivity in conn_measures)
    base = [p for p in parallelize]

    player_deck[node_pair] = [item for sublist in base for item in sublist]

dump(open(path_deck + f"player_deck_{woi}_sub{sub}.p", "wb"))

strategies = [mm_choice, mx_choice, av_choice, rn_choice]
strategy_ext = ["mm", "mx", "av", "rn"]

remap = interp1d([0,1],[1,100])

deck_remapped = {node_pair:[float(remap(val)) for val in player_deck[node_pair]] for node_pair in player_deck}

nodes = NODES[str(sub)]
resection = RESECTION[int(sub)]

Strategy_resection_score, Strategy_nonresection_score = [],[]

for i,strategy in enumerate(strategies):
    
    print(strategy_ext[i])

    test = {p:[] for p in players}

    rounds, turns = 100, 10
    for r in range(rounds):

        game = [Player(strategy, deck_remapped[p], p) for p in players]

        for j in range(10):
            n_cards = 24
            for player in game: player.n_in_hand = n_cards
            play(*game)

        scores = sorted([(player.name, player.score) for player in game], key=lambda x:x[1], reverse=True)

        top_score, fall = scores[0][1], 1
        for name, score in scores:
            if score==top_score: test[name].append(1); fall+=1
            elif score!=top_score: break
        for name,score in scores[fall:]: test[name].append(0)

    sorted_players = {(int(k[1:-1].split(", ")[0]),int(k[1:-1].split(",")[1])):v for k, v in sorted(test.items(), key=lambda item: sum(item[1]), reverse=True)}
    dump(sorted_players, open(path_scores + f"scores_{strategy_ext[i]}_{n_cards}cards_{rounds}rounds_{turns}turns_preseizure1_sub{sub}.p", "wb"))
    
    nonresection_score, resection_score = 0,0
    for k,v in sorted_players.items():
        if nodes[k[0]] in resection or nodes[k[1]] in resection:
            resection_score += sum(v)
        if nodes[k[0]] not in resection or nodes[k[1]] not in resection:
            nonresection_score += sum(v)

    print("Resection score:", resection_score/len(resection))
    print("Non-resection score:", nonresection_score/(len(nodes)-len(resection)))
