from random import randint

def av(of): return sum(of)/len(of)
def mm(of): return min(of)*max(of)/av(of)

class Player:
    def __init__(my, AI, lo=1, hi=10, hand=5):
        my.sets, my.logic, my.cards, my.score = (lo,hi,hand), AI, [], []
    def shuffle(hand):
        lo,hi,cards = hand.sets
        hand.cards = sorted(hand.cards+[randint(lo,hi) for c in range(cards-len(hand.cards))])
    def check(scores): return (1+scores.real)/(1+scores.best)
    def play(god, *among):
        best, other = max(god.cards), []
        for player in among: other+=player.cards
        if best>=max(other): return best
        else:                return min(god.cards)
    def card(draw): return draw.logic(draw)

def rn_choice(logic):
    return logic.cards.pop(randint(0,len(logic.cards)-1))

def av_choice(logic):
    lo,hi,_ = logic.sets
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
    winners = resolve([(god, god.play(*[other for other in game if other is not god])) for god in game])
    for player in winners: player.best += 1
    winners = resolve([(player, player.card()) for player in game])
    for player in winners: player.real += 1
    for player in game: player.score.append(player.check())

for what,mode in [('range 1-10',[(1,10)]), ('range 1-100',[(1,100)]), ('range 1-1000',[(1,1000)]), ('all ranges',[(1,10),(1,100),(1,1000)])]:
    print(what+'...')
    test, rounds = {rn_choice:[], av_choice:[], mx_choice:[], mm_choice:[]}, 1000
    while rounds:
        game = [Player(AI) for AI in (rn_choice, av_choice, mm_choice, mx_choice)]
        for lo,hi in mode:
            for cards in range(2,11):
                for player in game: player.sets = (lo,hi,cards)
                play(*game)
        scores = sorted([(player.logic,av(player.score)) for player in game], key=lambda x:x[1], reverse=True)
        for log,_ in scores[1:]: test[log].append(0)
        test[scores[0][0]].append(1)
        rounds -= 1
    print('guy (random    guy): {}'.format(sum(test[rn_choice])))
    print('ava (average   fan): {}'.format(sum(test[av_choice])))
    print('MAX (the   violent): {}'.format(sum(test[mx_choice])))
    print('max (game theorist): {}'.format(sum(test[mm_choice])))
    print('\n')