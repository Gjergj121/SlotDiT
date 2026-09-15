import numpy as np
import torch

def get_slots_stats(seq, masks):
    total_num_slots = len(torch.unique(masks))
    slot_dist = [len(torch.unique(m)) for m in masks]

    stats = {
            "total_num_slots": total_num_slots,
            "slot_dist": slot_dist,
            "max_num_slots": np.max(slot_dist),
            "min_num_slots": np.min(slot_dist)
        }
    return stats
