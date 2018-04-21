import unittest
from data.data_loader import VoiceDataset, VoiceData
from settings.hparam import hparam as hp
hp.set_hparam_yaml('test')


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.train_1_size = 4620
        self.train_data_1 = lambda: VoiceData(hp.test.test_data_path1)
        self.train_mean_1 = 1.1437469e-05
        self.delta = 1.0e-6

    def testVoiceDataLoading(self):
        train_data_1 = self.train_data_1()
        # mfcc
        train_data_1.mfcc()
        # phoneme
        train_data_1.phn()

    def testVoiceDataLoaderTrain(self):
        data_loader1 = VoiceDataset(mode='train')
        self.assertEqual(len(data_loader1), self.train_1_size)
        for ds in data_loader1:
            self.assertAlmostEqual(ds[0].mean(), self.train_mean_1, delta=self.delta)
            break


if __name__ == '__main__':
    unittest.main()
