from data.data_utils import get_wav_data, get_mfccs_phones, load_data
from settings.hparam import hparam as hp
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor


class VoiceData:

    def __init__(self, wav_file_path, mode='train', init_all=True):
        if mode not in ['train', 'test']:
            raise NotImplementedError('Mode %s is not implemented ! You can use train or test' % mode)
        self.wav_file_path = wav_file_path
        self.wav_data = get_wav_data(self.wav_file_path)
        self.mode = mode
        self.mfccs = None
        self.phns = None
        self.mels = None
        self.spec = None
        if init_all:
            self.init_data()

    @property
    def phn_file_path(self):
        if self.wav_file_path.endswith('arr'):
            return self.wav_file_path.replace('voice_all_arr/%s' % self.mode, 'voice_all').replace("WAV.wav.arr",
                                                                                                   "PHN").replace(
                "wav.arr", "PHN")
        else:
            return self.wav_file_path.replace("WAV.wav", "PHN").replace("wav", "PHN")

    @property
    def phn_length(self):
        return int(hp.default.duration // hp.default.frame_shift + 1)

    def init_data(self):
        is_random_crop = 'train' == self.mode
        # train 1 or test 1
        self.mfccs, self.phns = get_mfccs_phones(self.wav_data, self.phn_file_path,
                                                 random_crop=is_random_crop, trim=False)
        return self

    def mfcc(self):
        if self.mfccs is None:
            raise RuntimeError('Mfcc is not initialized !!!')
        return self.mfccs

    def phn(self):
        if self.phns is None:
            raise RuntimeError('Phoneme is not initialized !!!')
        return self.phns

    def wav(self):
        if self.wav_data is None:
            raise RuntimeError('Voice Data is not initialized !!!')
        return self.wav_data

    def mel(self):
        if self.mels is None:
            raise RuntimeError('Mel is not initialized !!!')
        return self.mels

    def spectrogram(self):
        if self.spec is None:
            raise RuntimeError('Spectrogram is not initialized !!!')
        return self.spec


class VoiceDataLoader(Dataset):

    def __init__(self, mode='train', data_split=-1.0, init_all=True):
        self.mode = mode
        self.wav_files = load_data(mode=mode, split=data_split)
        self.idx_list = list(range(len(self.wav_files)))
        self.init_all = init_all

    def __getitem__(self, idx):
        wav_file_path = self.wav_files[idx]
        voice_data = VoiceData(wav_file_path, self.mode, init_all=self.init_all)
        return FloatTensor(voice_data.mfcc()), LongTensor(voice_data.phn())

    def __len__(self):
        return len(self.idx_list)
