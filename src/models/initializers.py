import torch
import torch.nn as nn
from math import sqrt

from CONFIG import INITIALIZERS

ENCODER_RESOLUTION = (8, 14)

def get_initalizer(mode, slot_dim, num_slots, encoder_resolution=None):
    encoder_resolution = encoder_resolution if encoder_resolution is not None else ENCODER_RESOLUTION
    if mode not in INITIALIZERS:
        raise ValueError(f"Unknown initializer {mode = }. Available modes are {INITIALIZERS}")

    if mode == "Random":
        intializer = Random(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "Learned":
        intializer = Learned(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "LearnedRandom":
        intializer = LearnedRandom(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "Masks":
        raise NotImplementedError("'Masks' initialization is not supported...")
    elif mode == "CoM":
        intializer = CoordInit(slot_dim=slot_dim, num_slots=num_slots, mode="CoM")
    else:
        raise ValueError(f"UPSI, {mode = } should not have reached here...")

    return intializer

class Random(nn.Module):
    def __init__(self, slot_dim, num_slots):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

    def forward(self, batch_size, **kwargs):
        slots = torch.randn(batch_size, self.num_slots, self.slot_dim)
        return slots

class Learned(nn.Module):
    def __init__(self, slot_dim, num_slots):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots, -limit, limit)
        self.slots.requires_grad_()
        return

    def forward(self, batch_size, **kwargs):
        slots = self.slots.repeat(batch_size, 1, 1)
        return slots

class LearnedRandom(nn.Module):
    def __init__(self, slot_dim, num_slots):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_sigma, -limit, limit)
        return

    def forward(self, batch_size, **kwargs):
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_sigma.expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=self.slots_mu.device)
        return slots

class CoordInit(nn.Module):
    MODES = ["CoM"]
    MODE_REP = {
            "CoM": "com_coords"
        }
    IN_FEATS = {
            "CoM": 2
        }

    def __init__(self, slot_dim, num_slots, mode):
        assert mode in CoordInit.MODES, f"Unknown {mode = }. Use one of {CoordInit.MODES}"
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.mode = mode
        self.coord_encoder = nn.Sequential(
                nn.Linear(CoordInit.IN_FEATS[self.mode], 256),
                nn.ReLU(),
                nn.Linear(256, slot_dim),
            )
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))
        return

    def forward(self, batch_size, **kwargs):
        device = self.dummy_parameter.device
        rep_name = CoordInit.MODE_REP[self.mode]
        in_feats = CoordInit.IN_FEATS[self.mode]

        coords = kwargs.get(rep_name, None)
        if coords is None or coords.sum() == 0:
            raise ValueError(f"{self.mode} Initializer requires having '{rep_name}'...")
        if len(coords.shape) == 4:
            coords = coords[:, 0]
        coords = coords.to(device)

        num_coords = coords.shape[1]
        if num_coords > self.num_slots:
            raise ValueError(f"There shouldnt be more {num_coords = } than {self.num_slots = }! ")
        if num_coords < self.num_slots:
            remaining_masks = self.num_slots - num_coords
            pad_zeros = -1 * torch.ones((coords.shape[0], remaining_masks, in_feats), device=device)
            coords = torch.cat([coords, pad_zeros], dim=2)

        slots = self.coord_encoder(coords)
        return slots
