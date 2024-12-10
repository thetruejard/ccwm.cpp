
import torch
import torch.nn as nn
import subprocess

import ccwm_cpp_



if __name__ == '__main__':

    dtype = torch.float32

    l = nn.Linear(256, 128)

    torch.save({
        'cfg': {
            'size': 256
        },
        'weights': l.state_dict()
    }, 'test.pt')

    subprocess.run('python convert.py --in-file test.pt --out-file test.gguf', shell=True)

    t = ccwm_cpp_.Test('test.gguf', True)

