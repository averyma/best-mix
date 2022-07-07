import time
import os
from collections import defaultdict
import torch
from save_function import checkpoint_save

import matplotlib.pyplot as plt
import wandb

def saveCheckpoint(ckpt_dir, ckpt_name, epoch, state_dict, recorder, optimizer):

    checkpoint_save({"state_dict": state_dict,
                     "recorder": recorder,
                     "optimizer": optimizer,
                     "epoch": epoch},
                     ckpt_dir, ckpt_name+'.pth')

    print("SAVED CHECKPOINT")

class wandbLogger(object):
    def __init__(self, args, flush_sec=5):

        self.wandb_log = wandb.init(name=args.job_name,
                                    project=args.wandb_project,
                                    dir=args.root_dir,
                                    resume='allow',
                                    id=str(args.job_id),
                                    mode=args.wandb_mode)
        self.wandb_log.config.update(args)

#     def add_scalar(self, name, val, step, commit=True):
        # self.writer.add_scalar(name, val, step)

        # if "_itr" in name:
            # self.wandb_log.log({"iteration": step, name: float(val)}, commit=commit)
        # else:
#         self.wandb_log.log({"epoch": step, name: float(val)}, commit=commit)
        
    def add_scalar(self, saved_result):
        # self.writer.add_scalar(name, val, step)

        # if "_itr" in name:
            # self.wandb_log.log({"iteration": step, name: float(val)}, commit=commit)
        # else:
        self.wandb_log.log(saved_result)

