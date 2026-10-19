import pytest
import torch

from lerobot.policies.BPVAv31.modeling_bpvav31 import bp_align_loss


def test_mse_feature_token_sample_averaging():
    pred = torch.tensor([[[2., 0.], [100., 100.]], [[1., 1.], [3., 3.]]], requires_grad=True)
    mask = torch.tensor([[True, False], [True, True]])
    loss, count = bp_align_loss(pred, torch.zeros_like(pred), mask, "mse")
    assert loss.item() == pytest.approx(3.5)
    assert count.item() == 3
    loss.backward()
    assert torch.equal(pred.grad[0, 1], torch.zeros(2))


@pytest.mark.parametrize("kind", ["cosine", "mse"])
def test_no_valid_tokens_returns_zero(kind):
    pred = torch.ones(2, 3, 4, requires_grad=True)
    loss, count = bp_align_loss(pred, pred.detach(), torch.zeros(2, 3, dtype=torch.bool), kind)
    assert loss.item() == 0
    assert count.item() == 0
    loss.backward()


def test_cosine_default_and_raw_mse_differ():
    pred = torch.tensor([[[2., 0.]]])
    target = torch.tensor([[[1., 0.]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert bp_align_loss(pred, target, mask)[0].item() == pytest.approx(0)
    assert bp_align_loss(pred, target, mask, "mse")[0].item() == pytest.approx(0.5)


def test_unknown_loss_type_rejected():
    with pytest.raises(ValueError):
        bp_align_loss(None, None, None, "invalid")
