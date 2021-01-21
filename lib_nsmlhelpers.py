#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

from nmtlab.trainers import MTTrainer

def nsml_save_func(trainer, statedict, path=None):
    import nsml
    if path is not None:
        name = os.path.basename(path)
        name = name.replace(".pt", "")
        if "." in name:
            name = "model_{}".format(name.split(".")[-1])
        else:
            name = "model"
    else:
        name = "model"
    nsml.save("model")

def nsml_bind(trainer):
    import nsml
    if isinstance(trainer, MTTrainer):
        model = trainer.model()
    else:
        model = trainer
    def save(dir_path):
        state = {
            'model': model.state_dict(),
            # 'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(dir_path, 'model.pt'))

    def load(dir_path):
        state = torch.load(os.path.join(dir_path, 'model.pt'))
        model.load_state_dict(state['model'], strict=False)
        # if 'optimizer' in state and optimizer:
        #     optimizer.load_state_dict(state['optimizer'])
        #     print('optimizer loaded!')
        print('model loaded!')
    nsml.bind(save=save, load=load)
    if isinstance(trainer, MTTrainer):
        trainer.set_save_function(nsml_save_func)


def nsml_bind_full(trainer):
    import nsml
    if isinstance(trainer, MTTrainer):
        model = trainer.model()
    else:
        model = trainer
    def save(dir_path):
        state = trainer.state_dict()
        torch.save(state, os.path.join(dir_path, 'model.pt'))

    def load(dir_path):
        state = torch.load(os.path.join(dir_path, 'model.pt'))
        model.load_state_dict(state['model_state'])
        # if 'optimizer' in state and optimizer:
        #     optimizer.load_state_dict(state['optimizer'])
        #     print('optimizer loaded!')
        print('model loaded!')
    nsml.bind(save=save, load=load)
    if isinstance(trainer, MTTrainer):
        trainer.set_save_function(nsml_save_func)