## Implementation Speech Recognition Papers

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/AppleHolic/PytorchSR/LICENSE)

### Authors

1. appleholic (choiilji@gmail.com)
    - Occupation [KakaoBrain](http://www.kakaobrain.com/) AI Developer, june.one

### Implementations

#### sources

1. Completed to code training template.
2. TODOS:
    - evaluate
    - more papers
    
#### References

- phoneme classification

1. Prenet + CBHG in paper:
    - paper:  [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
    - PER : to be reported
2. Revising GRU :
    - paper: [Improving speech recognition by revising gated recurrent units](https://arxiv.org/abs/1710.00641)
    - Minimal GRU:
        - Implemented No Reset Gate GRU
        - TODOs:
            - Cuda Base Implementation
                - reference impl sample link : [https://github.com/chrischoy/pytorch-custom-cuda-tutorial](https://github.com/chrischoy/pytorch-custom-cuda-tutorial)
            - Recurrent Dropout

---

### Setup and Run

#### Environment
- python 3.6
- pytorch 0.3.1
- hyperparameters with yaml (in hparams folder)

#### Setup

```bash
$ pip install -r requirements.txt
```
    
#### Command

```bash
$ python run.py train --model cbhg
```

---


### Purposes:

1. Study *Speech Recognition Systems*
2. *Well define source code structure* in using pytorch 


