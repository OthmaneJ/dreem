import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SIGNALS_NAME = [
    "AbdoBelt",
    "AirFlow",
    "PPG",
    "ThorBelt",
    "Snoring",
    "SPO2",
    "C4A1",
    "O2A1",
]


def extract_events_from_binary_mask(binary_mask, fs=1):
    binary_mask = np.array([0] + binary_mask.tolist() + [0])
    diff_data = np.diff(binary_mask)
    starts = np.where(diff_data == 1)[0] / fs
    ends = np.where(diff_data == -1)[0] / fs

    assert len(starts) == len(ends)
    events = []
    for i, _ in enumerate(starts):
        events += [(starts[i], ends[i])]

    return events


def visualize_signal_and_event(X, mask, signals_name=SIGNALS_NAME, signal_freq=100):
    n_signal = X.shape[0]
    fig, axs = plt.subplots(n_signal, sharex=True,figsize=(10,15))

    events = extract_events_from_binary_mask(mask)
    for i in range(n_signal):
        axs[i].plot(np.arange(0, X[i].shape[0]) / signal_freq, X[i])
        axs[i].set_ylabel(signals_name[i])
        for elt in events:
            axs[i].axvspan(elt[0], elt[1], color='red', alpha=0.3)
    plt.xlim(0, X[0].shape[0]/ signal_freq)
    plt.show()


def visualise_index(idx, data_h5, masks, N_signals=8):
    x = data_h5['data'][idx, 2:]
    x = x.reshape(N_signals, -1)
    visualize_signal_and_event(x, np.array(masks[idx, 1:]))


if __name__ == "__main__":
    import h5py
    PATH_TO_TRAINING_DATA = "/content/drive/MyDrive/dreem_files/X_test.h5"
    PATH_TO_TRAINING_TARGET = "/content/drive/MyDrive/dreem/y_train.csv"
    h5_file = h5py.File(PATH_TO_TRAINING_DATA)
    mask = np.array(pd.read_csv(PATH_TO_TRAINING_TARGET))
    visualise_index(0, h5_file, mask)
    plt.plot()
