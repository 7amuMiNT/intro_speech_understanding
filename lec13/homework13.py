import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    回答:
import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    speech = np.asarray(speech, dtype=float)
    N = len(speech)

    if N < frame_length:
        A = np.zeros((0, order + 1), dtype=float)
        excitation = np.zeros((0, frame_length), dtype=float)
        return A, excitation

    nframes = 1 + (N - frame_length) // frame_skip
    A = np.zeros((nframes, order + 1), dtype=float)
    excitation = np.zeros((nframes, frame_length), dtype=float)

    win = np.hamming(frame_length)

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        a = librosa.lpc(frame * win, order=order)  # a[0]=1
        A[i, :] = a

        # residual e[n] = sum_{k=0..p} a[k] * x[n-k]
        e = np.zeros(frame_length, dtype=float)
        for n in range(frame_length):
            s = frame[n]
            kmax = min(order, n)
            for k in range(1, kmax + 1):
                s += a[k] * frame[n - k]
            e[n] = s

        excitation[i, :] = e

    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    e = np.asarray(e, dtype=float)
    A = np.asarray(A, dtype=float)
    nframes = A.shape[0]
    order = A.shape[1] - 1

    duration = nframes * frame_skip
    synthesis = np.zeros(duration, dtype=float)

    # keep filter memory across frames
    y_hist = np.zeros(order, dtype=float)

    for i in range(nframes):
        a = A[i, :]  # a[0]=1
        start = i * frame_skip
        end = start + frame_skip

        for n in range(start, end):
            yn = e[n] if n < len(e) else 0.0
            for k in range(1, order + 1):
                yn -= a[k] * (synthesis[n - k] if n - k >= 0 else y_hist[order - k])
            synthesis[n] = yn

        # update history (last 'order' samples up to current point)
        if order > 0:
            if end >= order:
                y_hist[:] = synthesis[end - order:end]
            else:
                y_hist[:end] = synthesis[:end]
                y_hist[end:] = 0.0

    return synthesis
def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    excitation = np.asarray(excitation, dtype=float)
    nframes, frame_length = excitation.shape
    T0 = int(T0)

    # gain from last frame_skip samples of each excitation frame
    gain = np.zeros(nframes, dtype=float)
    for i in range(nframes):
        tail = excitation[i, -frame_skip:] if frame_length >= frame_skip else excitation[i, :]
        gain[i] = np.sqrt(np.mean(tail**2)) if tail.size else 0.0

    # robot excitation: impulse train each frame, scaled by gain
    e_robot = np.zeros(nframes * frame_skip, dtype=float)
    if T0 > 0:
        for i in range(nframes):
            base = i * frame_skip
            for m in range(0, frame_skip, T0):
                e_robot[base + m] = -gain[i]

    return gain, e_robot
