import torch

from lingbot_video.transformer_lingbot_video import _packed_sdpa, _sdpa


def test_packed_sdpa_matches_independent_segments() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 7, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    packed = _packed_sdpa(q, k, v, cu_seqlens)
    expected = torch.cat(
        (
            _sdpa(q[:, :3], k[:, :3], v[:, :3]),
            _sdpa(q[:, 3:], k[:, 3:], v[:, 3:]),
        ),
        dim=1,
    )
    torch.testing.assert_close(packed, expected)


def test_packed_sdpa_does_not_mix_samples() -> None:
    torch.manual_seed(11)
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

    baseline = _packed_sdpa(q, k, v, cu_seqlens)
    changed_v = v.clone()
    changed_v[:, 4:] += 10_000
    changed = _packed_sdpa(q, k, changed_v, cu_seqlens)
    torch.testing.assert_close(changed[:, :4], baseline[:, :4])


def test_packed_sdpa_is_differentiable() -> None:
    q = torch.randn(1, 5, 2, 4, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    output = _packed_sdpa(
        q,
        k,
        v,
        torch.tensor([0, 2, 5], dtype=torch.int32),
    )
    output.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
