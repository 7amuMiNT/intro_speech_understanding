import numpy as np
import torch, torch.nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    
    '''
   # features: pre-emphasis + spectrogram (4ms/2ms) + keep low-frequency half
    fl_feat = int(np.round(0.004 * Fs))  # 4 ms
    st_feat = int(np.round(0.002 * Fs))  # 2 ms

    wp = preemphasis(waveform, a=0.97)
    frames = waveform_to_frames(wp, fl_feat, st_feat)
    mstft = frames_to_mstft(frames)
    spec_db = mstft_to_spectrogram(mstft)

    nfeats = fl_feat // 2
    features = spec_db[:, :nfeats].astype(np.float32)

    # labels: VAD (25ms/10ms), segment ids, then repeat each label five times (10ms -> 2ms)
    vad_labels = vad_frame_labels(waveform, Fs)  # length ~ every 10ms
    labels_rep = np.repeat(vad_labels, 5)        # now ~ every 2ms

    # match feature frame count
    NFRAMES = features.shape[0]
    labels = np.zeros((NFRAMES,), dtype=int)
    L = min(NFRAMES, len(labels_rep))
    labels[:L] = labels_rep[:L]
    return features, labels

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step.
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
     X = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    nfeats = X.shape[1]
    nlabels = int(y.max().item()) + 1 if y.numel() else 1

    model = nn.Sequential(
        nn.LayerNorm(nfeats),
        nn.Linear(nfeats, nlabels)
    )

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    lossvalues = np.zeros((int(iterations),), dtype=np.float32)
    model.train()
    for it in range(int(iterations)):
        optim.zero_grad()
        logits = model(X)              # (NFRAMES, NLABELS)
        loss = criterion(logits, y)    # expects class indices
        loss.backward()
        optim.step()
        lossvalues[it] = float(loss.item())

    return model, lossvalues
def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    X = torch.from_numpy(np.asarray(features, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probabilities
