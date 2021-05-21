import pytest


torch = pytest.importorskip("torch")

from src.models import UNet
from src.utils import count_parameters, resolve_device, set_seed


def test_set_seed_reproducibility_for_torch():
    set_seed(123)
    first = torch.rand(3)
    set_seed(123)
    second = torch.rand(3)
    assert torch.allclose(first, second)


def test_count_parameters_trainable_subset():
    model = UNet(in_channels=1, out_channels=2)
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)

    assert trainable <= total
    assert trainable > 0


def test_resolve_device_returns_torch_device():
    device = resolve_device(prefer_cuda=False)
    assert isinstance(device, torch.device)
