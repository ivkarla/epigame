# pylint: disable=no-self-argument, no-member

from numpy import correlate, average, array, angle, mean, sign, exp, zeros, abs, unwrap
from numpy.linalg import norm
from awc import error
from scipy.signal import coherence, hilbert, csd
from matplotlib.mlab import cohere
from itertools import combinations
from data_legacy import butter_filter
from sklearn.preprocessing import normalize

def phaselock(signal1, signal2):
    '''phase locking value between signal1 and signal2;       
       NOTE: signals must be bandpass filtered'''
    sig1_hil = hilbert(signal1)                          
    sig2_hil = hilbert(signal2)
    phase1 = angle(sig1_hil)                           
    phase2 = angle(sig2_hil)
    phase_dif = phase1-phase2                             
    plv = abs(mean(exp(complex(0,1)*phase_dif)))    
    return plv

def phaselag(signal1, signal2):
    '''phase lag index between signal1 and signal2'''
    sig1_hil = hilbert(signal1)                 
    sig2_hil = hilbert(signal2)
    phase1 = angle(sig1_hil)                 
    phase2 = angle(sig2_hil)
    phase_dif = phase1-phase2                   
    pli = abs(mean(sign(phase_dif)))     
    return pli

def spectral_coherence(signal1, signal2, fs, imag=False):
    '''spectral coherence between signal1 and signal2;
    returns the real part of the complex value when imag is False, otherwise returns the imaginary part'''
    Pxy = csd(signal1,signal2,fs=fs, scaling='spectrum')[1] 
    Pxx = csd(signal1,signal1,fs=fs, scaling='spectrum')[1]
    Pyy = csd(signal2,signal2,fs=fs, scaling='spectrum')[1]
    if imag: return average((Pxy.imag)**2/(Pxx*Pyy))     
    elif not imag: return average(abs(Pxy)**2/(Pxx*Pyy))

def cross_correlation(signal1, signal2):
    '''cross correlation between signal1 and signal2'''                                                
    return correlate(signal1, signal2, mode="valid")

def PAC(signal1, signal2, fs):
    '''low frequency phase-high frequency amplitude phase coupling between signal1 and signal2;
    low frequency = delta (1-4Hz)
    high frequency = low gamma (30-70Hz)'''   
    low = butter_filter(signal1,1,4,fs) 
    high = butter_filter(signal2,30,70,fs) 
    low_hil = hilbert(low)
    low_phase_angle = unwrap(angle(low_hil))   
    high_env_hil = hilbert(abs(hilbert(high)))
    high_phase_angle = unwrap(angle(high_env_hil))
    phase_dif = low_phase_angle - high_phase_angle 
    plv = abs(mean(exp(complex(0,1)*phase_dif)))
    return plv

def connectivity_analysis(epochs, method, dtail=False, **opts):
    '''returns a connectivity matrix N×N (N - number of nodes), using a method for each node pair;
    return a triangular matrix when dtail is False, otherwise return a regular matrix'''
    print('Connectivity Analysis: '+str(method).split()[1])
    result = [] 
    for i,e in enumerate(epochs):    
        mat = zeros((len(e),len(e)))                                    
        nid, pairs = list(range(len(e))), []
        for a in range(len(nid)):                             
            if dtail:
                for b in range(len(nid)): pairs.append((a,b))
            else:
                for b in range(a, len(nid)): pairs.append((a,b))                                       
        for pair in pairs:                                                       
            mat[pair[0],pair[1]] = method(e[pair[0]], e[pair[1]], **opts)
        result.append(mat)     
        print('{}: completed '.format(i), end='\n')                                                                                       
    return result

def PEC(nse, n):
    '''prediction error connectivity;
    returns a connectivity matrix N×N (N-number of nodes)'''
    print('{}: '.format(n), end='')
    return array(error(nse, 2)[1])
    
    
    