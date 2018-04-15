import fire


class Runner:

    def train(self):
        pass

    def test(self):
        raise NotImplementedError('Test mode is not implemented!')

    def eval(self):
        raise NotImplementedError('Evaluation mode is not implemented!')


if __name__ == '__main__':
    fire.Fire(Runner)
