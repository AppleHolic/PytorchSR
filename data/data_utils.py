import glob
import numpy as np
import librosa
import pyarrow as pa
from sklearn.model_selection import train_test_split
from .audio import preemphasis, amp_to_db
from ..settings.hparam import hparam as hp


PHNS = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def get_wav_data(wav_file, sr=None):
    if wav_file.endswith('arr'):
        wav = load_arrow_file(wav_file)
    else:
        sr = sr if sr else hp.default.sr
        wav, _ = librosa.load(wav_file, sr=sr)
    return wav


def get_mfccs_phones(wav_data, phn_file, trim=False, random_crop=True):
    mfccs, _, _ = _get_mfcc_log_spec_and_log_mel_spec(wav_data, hp.default.preemphasis, hp.default.n_fft,
                                                      hp.default.win_length,
                                                      hp.default.hop_length)
    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    # phn_file = wav_file.replace("WAV.wav", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp.default.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1
    # Random crop
    if random_crop:
        start = np.random.choice(range(int(np.maximum(1, len(mfccs) - n_timesteps))), 1)[0]
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

    return mfccs, phns


def get_mfccs_and_spectrogram(wav_data, win_length, hop_length, trim=True, duration=None, random_crop=False):
    # Trim
    if trim:
        wav_data, _ = librosa.effects.trim(wav_data, frame_length=win_length, hop_length=hop_length)
    if random_crop:
        wav_data = wav_random_crop(wav_data, hp.default.sr, duration)
    # Padding or crop
    if duration:
        length = hp.default.sr * duration
        wav_data = librosa.util.fix_length(wav_data, length)
    return _get_mfcc_and_spec(wav_data, hp.default.preemphasis, hp.default.n_fft, win_length, hop_length)


def _get_mfcc_log_spec_and_log_mel_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):
    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = amp_to_db(mag)
    mel_db = amp_to_db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    # Quantization
    # bins = np.linspace(0, 1, hp.default.quantize_db)
    # mag_db = np.digitize(mag_db, bins)
    # mel_db = np.digitize(mel_db, bins)

    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = amp_to_db(mag)
    mel_db = amp_to_db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


def _get_zero_padded(list_of_arrays):
    '''
    :param list_of_arrays
    :return: zero padded array
    '''
    batch = []
    max_len = 0
    for d in list_of_arrays:
        max_len = max(len(d), max_len)
    for d in list_of_arrays:
        num_pad = max_len - len(d)
        pad_width = [(0, num_pad)]
        for _ in range(d.ndim - 1):
            pad_width.append((0, 0))
        pad_width = tuple(pad_width)
        padded = np.pad(d, pad_width=pad_width, mode="constant", constant_values=0)
        batch.append(padded)
    return np.array(batch)


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(PHNS)}
    idx2phn = {idx: phn for idx, phn in enumerate(PHNS)}

    return phn2idx, idx2phn


def load_data(mode, split=-1.0):
    """
    load file path list from hparams
    :param mode: runtime mode
    :param split: the percentage of training set. if the value is between 0.0 and 1.0,
    function returns splitted list using mode
    :return:
    """
    file_path = getattr(hp, mode).data_path.replace('voice_all', 'voice_all_arrow/%s' % mode).replace('.wav', '.wav.arr')
    wav_files = glob.glob(file_path)
    if len(wav_files) == 0:
        wav_files = glob.glob(getattr(hp, mode).data_path)
    if split and 0.0 < split < 1.0:
        train_wav_files, test_wav_files = train_test_split(wav_files, train_size=split, random_state=1234)
        if 'train' in mode:
            wav_files = train_wav_files
        elif 'test' in mode:
            wav_files = test_wav_files
        else:
            NotImplementedError('Other mode are not implemented in split mode! (mode: %s, split: %.4f)' % (mode, split))
    return wav_files


# serialize data function
serialize_data = lambda arr: pa.serialize(arr).to_buffer().to_pybytes()
deserialize_data = lambda bin_data: pa.deserialize(bin_data)


def load_arrow_file(file_path):
    with open(file_path, 'rb') as rb:
        return deserialize_data(rb.read())


def normalize_0_1(values, max, min):
    normalized = np.clip((values - min) / (max - min), 0, 1)
    return normalized


def denormalize_0_1(normalized, max, min):
    values = np.clip(normalized, 0, 1) * (max - min) + min
    return values
