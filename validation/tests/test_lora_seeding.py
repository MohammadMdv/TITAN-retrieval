"""Seed k's LoRA adapter must be a pure function of k."""
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import finetune_bracs_lora as fl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _peft():
    return get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=["fc1", "fc2"],
                                             bias="none"))


def _adapter_state(m):
    return {n: p.detach().clone() for n, p in m.named_parameters() if "lora_" in n}


def test_reset_is_deterministic_given_the_seed():
    """Same seed -> same adapter, on two independently constructed models."""
    a, b = _peft(), _peft()
    torch.manual_seed(7); fl.reset_lora(a)
    torch.manual_seed(7); fl.reset_lora(b)
    sa, sb = _adapter_state(a), _adapter_state(b)
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_different_seeds_give_different_adapters():
    a, b = _peft(), _peft()
    torch.manual_seed(0); fl.reset_lora(a)
    torch.manual_seed(1); fl.reset_lora(b)
    diff = [k for k in _adapter_state(a)
            if "lora_A" in k and not torch.equal(_adapter_state(a)[k], _adapter_state(b)[k])]
    assert diff, "seeds 0 and 1 produced identical lora_A -- reset is not drawing from the RNG"


def test_seed_k_does_not_depend_on_earlier_seeds():
    """The real bug: seed 1's adapter must be the same whether or not seed 0 ran first."""
    m = _peft()
    torch.manual_seed(1); fl.reset_lora(m)
    alone = _adapter_state(m)

    m2 = _peft()
    torch.manual_seed(0); fl.reset_lora(m2)
    _ = torch.randn(1000)                       # stand in for seed 0's training consuming RNG
    torch.manual_seed(1); fl.reset_lora(m2)     # seed 1 re-seeds first -> must be unaffected
    after = _adapter_state(m2)

    for k in alone:
        assert torch.equal(alone[k], after[k]), k


def test_lora_B_starts_at_zero():
    """B=0 means the adapter is a no-op at init: an untrained LoRA == the frozen encoder."""
    m = _peft()
    torch.manual_seed(3); fl.reset_lora(m)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            assert torch.count_nonzero(p) == 0, n
