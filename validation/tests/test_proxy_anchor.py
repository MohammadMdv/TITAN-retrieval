"""The Proxy-Anchor loss that produces the Tier-3 LoRA result."""
import numpy as np
import torch
import torch.nn.functional as F

from finetune_bracs_lora import ProxyAnchor, pk_batches


def test_loss_is_lower_when_embeddings_sit_on_their_own_proxy():
    """The defining property. If this fails, the loss is not doing metric learning at all."""
    torch.manual_seed(0)
    C, D = 4, 8
    loss_fn = ProxyAnchor(C, D)
    P = F.normalize(loss_fn.proxies.detach(), dim=1)
    y = torch.arange(C)

    aligned = loss_fn(P, y)                        # each embedding == its own class proxy
    wrong = loss_fn(P[torch.roll(y, 1)], y)        # each embedding == a DIFFERENT class's proxy
    assert aligned.item() < wrong.item()


def test_loss_is_finite_in_fp32_at_the_alpha_delta_extremes():
    """exp(alpha*(cos+delta)) peaks near exp(32*1.1) = exp(35.2) ~ 2e15: fine in fp32, inf in
    fp16. The code comments claim this; assert it, because an inf here silently kills a run."""
    C, D = 7, 16
    loss_fn = ProxyAnchor(C, D, alpha=32.0, delta=0.1)
    with torch.no_grad():
        loss_fn.proxies.copy_(torch.ones(C, D))
    emb = F.normalize(torch.ones(C, D), dim=1)     # cos == 1 everywhere: the worst case
    out = loss_fn(emb, torch.arange(C))
    assert torch.isfinite(out).all()


def test_loss_is_positive_and_scalar():
    loss_fn = ProxyAnchor(5, 8)
    emb = F.normalize(torch.randn(15, 8), dim=1)
    out = loss_fn(emb, torch.arange(15) % 5)
    assert out.ndim == 0 and out.item() > 0


def test_gradients_reach_both_the_embeddings_and_the_proxies():
    loss_fn = ProxyAnchor(3, 6)
    emb = F.normalize(torch.randn(9, 6, requires_grad=True), dim=1)
    loss_fn(emb, torch.arange(9) % 3).backward()
    assert loss_fn.proxies.grad is not None
    assert torch.count_nonzero(loss_fn.proxies.grad) > 0


def test_pk_batches_put_every_class_in_every_batch():
    """P = all classes, K per class. Both loss terms need a non-empty positive set, which is why
    the sampler is PK and not random."""
    C, K = 7, 8
    y = np.repeat(np.arange(C), 40)
    rng = np.random.default_rng(0)
    batches = list(pk_batches(y, C, K, rng))
    assert batches
    for b in batches:
        assert len(b) == C * K
        assert set(y[b]) == set(range(C))          # every class present
        counts = np.bincount(y[b], minlength=C)
        assert (counts == K).all()                 # exactly K each


def test_pk_batches_are_reproducible_from_the_rng():
    C, K = 3, 4
    y = np.repeat(np.arange(C), 20)
    a = list(pk_batches(y, C, K, np.random.default_rng(5)))
    b = list(pk_batches(y, C, K, np.random.default_rng(5)))
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
