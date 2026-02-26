import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    frame_length (scalar) - length of the frame, in samples
    step (scalar) - step size, in samples
    
    @returns:
    frames (np.ndarray((num_frames, frame_length))) - waveform chopped into frames
       frames[m/step,n] = waveform[m+n] only for m = integer multiple of step
    '''
    waveform = np.asarray(waveform)
    N = len(waveform)

    if frame_length <= 0 or step <= 0:
        raise ValueError("frame_length and step must be positive")

    # number of frames so that each frame fits entirely in waveform
    if N < frame_length:
        return np.zeros((0, frame_length), dtype=waveform.dtype)

    num_frames = 1 + (N - frame_length) // step
    frames = np.zeros((num_frames, frame_length), dtype=waveform.dtype)

    for i in range(num_frames):
        m = i * step
        frames[i, :] = waveform[m:m + frame_length]

    return frames
def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    
    @params:
    frames (np.ndarray((num_frames, frame_length))) - the speech samples
    
    @returns:
    mstft (np.ndarray((num_frames, frame_length))) - the magnitude short-time Fourier transform
    '''
    frames = np.asarray(frames)
    # FFT along each row (frame)
    mstft = np.abs(np.fft.fft(frames, axis=1))
    return mstft
def mstft_to_spectrogram(mstft):
    '''
    Convert max(0.001*amax(mstft), mstft) to decibels.
    
    @params:
    stft (np.ndarray((num_frames, frame_length))) - magnitude short-time Fourier transform
    
    @returns:
    spectrogram (np.ndarray((num_frames, frame_length)) - spectrogram 
    
    The spectrogram should be expressed in decibels (20*log10(mstft)).
    np.amin(spectrogram) should be no smaller than np.amax(spectrogram)-60
    '''
   mstft = np.asarray(mstft)
    amax = np.amax(mstft) if mstft.size else 0.0

    # floor at 0.001*amax (i.e., -60 dB relative to peak), and also avoid log(0)
    floor = max(0.001 * amax, 1e-12)
    mstft_clipped = np.maximum(mstft, floor)

    spectrogram = 20.0 * np.log10(mstft_clipped)

    return spectrogram

