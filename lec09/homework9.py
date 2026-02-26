import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    frame_length = int(np.round(0.025 * Fs))  # 25 ms
    step = int(np.round(0.010 * Fs))          # 10 ms

    frames = waveform_to_frames(waveform, frame_length, step)
    if frames.shape[0] == 0:
        return []

    energy = np.sum(frames.astype(float) ** 2, axis=1)
    thr = 0.1 * np.max(energy)
    active = energy > thr

    segments = []
    N = len(waveform)

    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        start = i * step
        j = i
        while j < len(active) and active[j]:
            j += 1
        end = (j - 1) * step + frame_length
        end = min(end, N)
        segments.append(np.asarray(waveform[start:end]))
        i = j

    return segments
def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []
    frame_length = int(np.round(0.004 * Fs))  # 4 ms
    step = int(np.round(0.002 * Fs))          # 2 ms

    for seg in segments:
        segp = _preemphasis(seg, a=0.97)
        frames = waveform_to_frames(segp, frame_length, step)
        if frames.shape[0] == 0:
            models.append(np.zeros(frame_length // 2, dtype=float))
            continue

        mstft = frames_to_mstft(frames)
        spec = mstft_to_spectrogram(mstft)

        half = frame_length // 2  # low-frequency half
        low = spec[:, :half]
        model = np.mean(low, axis=0)
        models.append(model)

    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K), dtype=float)

    for y in range(Y):
        for k in range(K):
            sims[y, k] = _cosine_sim(models[y], test_models[k])

    test_outputs = []
    for k in range(K):
        best = int(np.argmax(sims[:, k])) if Y > 0 else 0
        test_outputs.append(labels[best] if labels else "")

    return sims, test_outputs


