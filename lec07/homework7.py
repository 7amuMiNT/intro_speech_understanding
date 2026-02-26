import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    # half-second duration
    N = int(np.round(0.5 * Fs))
    n = np.arange(N)

    # major third: +4 semitones, major fifth: +7 semitones
    f3 = f * (2 ** (4/12))
    f5 = f * (2 ** (7/12))

    # sum of three cosines (optionally scaled to avoid clipping)
    x = (np.cos(2*np.pi*f*n/Fs) +
         np.cos(2*np.pi*f3*n/Fs) +
         np.cos(2*np.pi*f5*n/Fs)) / 3.0
    return x
def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    n = np.arange(N)
    k = np.arange(N)
    W = np.exp(-1j * 2*np.pi * np.outer(k, n) / N).astype(complex)
    return W
def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    x = np.asarray(x)
    N = len(x)

    W = dft_matrix(N)
    X = W @ x
    mag = np.abs(X)

    # use only nonnegative frequencies (unique bins for real signals)
    kmax = N // 2
    mag_pos = mag[:kmax+1]

    # pick 3 largest bins
    topk = np.argsort(mag_pos)[-3:]
    freqs = np.sort(topk * Fs / N)

    return float(freqs[0]), float(freqs[1]), float(freqs[2])
