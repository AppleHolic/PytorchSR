## Implementation Speech Recognition Papers

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/AppleHolic/PytorchSR/LICENSE)

---

### Authors

1. appleholic (choiilji@gmail.com)
    - [KakaoBrain](http://www.kakaobrain.com/) AI Developer, june.one

### Implementations

#### sources

1. Higher Level Implementation with *overrided pytorch utils*
2. WILL BE UPDATED
    - Checkpoint plugin
    - Tensorboard plugin 
    
#### Phoneme Classification

1. Prenet + CBHG in paper:
    - [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
2. Will be implemented paper : 
    - [Phone recognition with hierarchical
convolutional deep maxout networks](https://link.springer.com/content/pdf/10.1186%2Fs13636-015-0068-3.pdf)
    
---

### Setup and Run

#### Environment
- python 3.6
- pytorch 0.3.1
- hyperparameters with yaml (in hparams folder)

#### Setup
```
$ pip install -r requirements.txt
```
    
#### Command

```bash
$ python run.py --model cbhg
```

---


### Purposes:

1. Study *Speech Recognition Systems*
2. *Well define source code structure* in using pytorch 


