# pylint: disable=no-self-argument, no-member

from src.eeg import EEG, SET, STEp, epoch, secs, ms, np, struct, preprocess
from src.core import REc as rec
from src.connectivity import preprocess, connectivity_analysis, phaselock, phaselag, spectral_coherence, PEC, cross_correlation, PAC
from src.game import analyze, enlist, check_until

from itertools import combinations
import os

source = "../data/" 
preprocessed = "../results/preprocessed/"
out = "../data/results/EN/"

"""INITIALIZATION: method and frequency band of choice"""  

window = input("Time window:\n 1. Non-seizure (baseline)\n 2. Pre-seizure\n 3. Transition to seizure\n 4. Seizure\n Indicate a number: ")
method_idx = input("Connectivity method:\n 1. PEC\n 2. Spectral Coherence\n 3. Phase Lock Value\n 4. Phase-Lag Index\n 5. Cross-correlation\n 6. Phase-amplitude coupling\n Indicate a number: ")
ext = ""
if "2" == method_idx: 
    im = input("Imaginary part (Y/N): ").upper()
    if im == "Y": imag,ext = True,"I"
    elif im == "N": imag,ext = False,"R"
bands, Bands = input("Filter the signal: Y/N ").upper(), False
if bands=="N": bands = "w"
elif bands=="Y": 
    Bands = True
    mn = int(input("Set band range min: "))
    mx = int(input("Set band range max: "))
    bands = (mn,mx)
dict_methods = {'1':PEC, '2':spectral_coherence, '3':phaselock, '4':phaselag, '5':cross_correlation, '6':PAC}
method_code = {'1':"PEC", '2':"SC_", '3':"PLV", '4':"PLI", '5':"CC", '6':"PAC"}   
window_code = {'1':"baseline", '2':"preseizure", '3':"transition", '4':"seizure"}
files = list(os.listdir(source))
subjects = [fname[0:3] for fname in files]
limit = 500
step, span = 500, 1000            #to increase stats power, decrease step to 250 or 100, modyfing their ratio accordingly (span/step)
ratio = span/step
A, B = 'S', 'E'                   
items = 30                        
retry_ratio = 0                   #at the beginning we thought to use a percentage of all found networks, but that leads to an exponential increase that prooves to be unmanageable
max_netn = 10                     #max number of nodes per network (10 is to avoid too long processing times and also a statistical failure, if networks are too big they will ultimately cover all the explored area)


"""ANALYSIS PIPELINE"""  

for subid in subjects:  
    
    """LOADING DATA""" 

    print('processing {}'.format(subid), end='...')
    name = ''
    for filename in files:
        if filename.startswith(subid): name = filename; break        
    eeg = EEG.from_file(source+name, epoch(ms(step), ms(span)))
    print(subid, "\n sampling: ", eeg.fs)
    if eeg.fs == 512: limit = 512
    else: limit = 500

    """PREPROCESSING""" 

    SET(eeg, _as='N')
    SET(eeg, 'EEG inicio', 'S')
    SET(eeg, 'EEG fin', 'E', epoch.END)
    eeg.optimize()
    eeg.remap()
    units = int((eeg.notes['EEG fin'][0].time - eeg.notes['EEG inicio'][0].time)*ratio) 
    
    '''tagging epochs relative to seizure duration'''
    if window == "1":
        pre = int(round(units))
        eeg.tag(('S','E'), S=range(-pre,0,1), E=range(0,-units,-1))
    elif window == "2":
        pre = int(round(units*.6)) 
        eeg.tag(('S', 'E'), S=range(-pre,0,1), E=range(0,-units,-1))
    elif window == "3" or window == "4":
        pre = int(round(units*.3))
        eeg.tag(('S', 'E'), S=range(-pre,pre,1), E=range(0,-units,-1)) 

    a, ai = eeg.sample.get(A, items)   
    b, bi = eeg.sample.get(B, items)   
    i = ai+bi                         
    x = a + b                         
    y = [0]*items + [1]*items          
    
    esppe = [preprocess(eeg, ep, limit) for i,ep in enumerate(x)] 
    print("resampled: ", esppe[0].shape)

    '''filtering frequency bands'''
    fesppe = []
    if Bands: fesppe = [band(e, bands, esppe[0].shape[1]) for e in esppe]
    elif not Bands: fesppe = esppe

    '''connectivity analysis'''
    result = struct(x=np.array(x), y=np.array(y), i=np.array(i))
    if dict_methods[method_idx] == spectral_coherence:
        result._set(X = connectivity_analysis(fesppe,spectral_coherence,fs=fesppe[0].shape[1],imag=imag))
    elif dict_methods[method_idx] == PEC: 
        result._set(X = [PEC(ep,i+1) for i,ep in enumerate(fesppe)])
    elif dict_methods[method_idx] == phaselock:
        result._set(X = connectivity_analysis(fesppe, phaselock))
    elif dict_methods[method_idx] == phaselag:
        result._set(X = connectivity_analysis(fesppe, phaselag))
    elif dict_methods[method_idx] == cross_correlation:
        result._set(X = connectivity_analysis(fesppe, cross_correlation))
    elif dict_methods[method_idx] == PAC:
        result._set(X = connectivity_analysis(fesppe, PAC, True, fesppe[0].shape[1]))

    '''saving preprocessed data'''
    if Bands: rec(result).save(preprocessed+"{}/".format(method_code[method_idx]+ext)+"{}/".format(str(bands).replace(" ",""))+'.'.join(['-'.join([subid,window_code[window]]),'prep']))
    elif not Bands: rec(result).save(preprocessed+"{}/".format(method_code[method_idx]+ext)+'.'.join(['-'.join([subid,window_code[window]]),'prep']))

    """ANALYSIS"""
    
    '''load preprocessed data'''
    nid, nodes = list(range(len(eeg.axes.region))), list(eeg.axes.region) 
    nxn, base, ftype, rtype = combinations(nid, 2), [], '.prep', '.res'  
    AB, pid = (A+B).lower(), subid
    pid = '-'.join([pid,AB])
    if Bands: prep_path=preprocessed+"{}/".format(method_code[method_idx]+ext)+"{}/".format(str(bands).replace(" ",""))+'.'.join(['-'.join([subid,window_code[window]]),'prep'])
    elif not Bands: prep_path=preprocessed+"{}/".format(method_code[method_idx]+ext)+'.'.join(['-'.join([subid,window_code[window]]),'prep'])

    pdata = rec.load(prep_path).data
    print('processing base combinations...', end=' ')
    for pair in nxn: base.append(enlist(pair, nodes, analyze(pdata, pair))) 
    print('{} done'.format(len(base)))
    base.sort(key=lambda x:x[-1], reverse=True)
    print('best hub: {}'.format(base[0][1]), end='; ')
    best, netn, sets, nets = base[0][-1], 3, base[:], []

    while netn<=max_netn:
        print('checking {} nodes'.format(netn), end='... ')
        head, tests = check_until(sets),0
        for hub in sets[:head if head>0 else 1]:
            for node in nid:
                if node not in hub[0]:
                    test = hub[0]+(node,)
                    nets.append(enlist(test, nodes, analyze(pdata, test)))
                    tests += 1
        print('{} done'.format(tests))
        nets.sort(key=lambda x:x[-1], reverse=True)
        print('best net: {}'.format(nets[0][1]), end='')
        if nets[0][-1]>=best:
            if netn<max_netn:
                best = nets[0][-1]
                sets = nets[:]
                nets = []
                print(';', end=' ')                
            netn += 1
        else: break
    selected = sorted(set([t for n in nets[:check_until(nets)] for t in n[1].split('<->')]))
    print('\nselected nodes: {}/{}.'.format(', '.join(selected), len(selected)))
    
    '''saving analysis result'''
    if Bands: rec(struct(base=base, nets=nets, nodes=selected)).save(out+"{}/".format(method_code[method_idx]+ext)+"{}/".format(str(bands).replace(" ",""))+'-'.join([subid,window_code[window]])+rtype)
    elif not Bands: rec(struct(base=base, nets=nets, nodes=selected)).save(out+"{}/".format(method_code[method_idx]+ext)+'-'.join([subid,window_code[window]])+rtype)
