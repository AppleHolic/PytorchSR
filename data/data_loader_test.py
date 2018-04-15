import unittest
from .data_loader import VoiceDataLoader, VoiceData
from ..settings.hparam import hparam
hparam.set_hparam_yaml('test')


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.train_1_size = 4620
        self.train_2_size = 1132
        self.train_data_1 = lambda: VoiceData('/data/public/rw/datasets/voice_all/timit/TIMIT/TRAIN/DR5/FDMY0/SA1.wav')
        self.train_data_2 = lambda: VoiceData('/data/public/rw/datasets/voice_all/arctic/slt/arctic_a0001.wav', mode='train2')
        self.train_mean_1 = 1.1437469e-05
        self.train_mean_2 = -2.7451038e-5
        self.delta = 1.0e-6

    def testVoiceDataLoading(self):
        train_data_1 = self.train_data_1()
        train_data_2 = self.train_data_2()
        # mfcc
        train_data_1.mfcc()
        train_data_2.mfcc()
        # spectrogram
        train_data_2.spectrogram()
        # phoneme
        train_data_1.phn()

    def testVoiceDataLoaderTrain1(self):
        data_loader1 = VoiceDataLoader(mode='train1', is_shuffle=False)
        self.assertEqual(data_loader1.size(), self.train_1_size)
        for ds in data_loader1.get_data():
            self.assertAlmostEqual(ds[0].wav().mean(), self.train_mean_1, delta=self.delta)
            break

    def testVoiceDataLoaderTrain2(self):
        data_loader2 = VoiceDataLoader(mode='train2', is_shuffle=False)
        self.assertEqual(data_loader2.size(), self.train_2_size)
        for ds in data_loader2.get_data():
            self.assertAlmostEqual(ds[0].wav().mean(), self.train_mean_2, delta=self.delta)
            break


if __name__ == '__main__':
    unittest.main()
