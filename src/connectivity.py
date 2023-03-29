# pylint: disable=no-self-argument, no-member

from numpy import correlate, average, array, angle, mean, sign, exp, zeros, abs, unwrap
from src.awc import error
from scipy.signal import hilbert, csd
from src.data_legacy import butter_filter, notch, dwindle, upsample
from joblib import Parallel, delayed

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

def PAC(signal1, signal2, fs=500):
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

def connectivity_analysis(epochs, method, dtail=False, **opts):
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
    
    
    