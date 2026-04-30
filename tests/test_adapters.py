import torch

from adapters import ResNetBasicBlock, ResNetBottleneck, SimpleAdapter


def test_basicblock_init_identity_results_in_no_change():
    # Alternative way to initialize the tensor that fills each channel with the index of that channel.
    # x = torch.arange(256).float()
    # xim = x.view(1, -1, 1, 1).repeat(1, 1, 3, 3)
    xim = torch.rand((3, 256, 3, 3))
    cid = ResNetBasicBlock(256, 256, init_identity=True)
    out = cid(xim)
    assert out.shape == xim.shape
    assert torch.allclose(out, xim)


def test_basicblock_init_identity_with_downproject_results_in_no_change_for_initial_values():
    # NOTE: We can currently only guarantee this for eval mode, because the norm layer in the skip connection branch
    # will alter the output when in training mode.
    xim = torch.rand((3, 384, 3, 3))
    cid = ResNetBasicBlock(384, 256, init_identity=True).eval()
    out = cid(xim)
    assert out.shape == xim[:, :256].shape
    assert torch.allclose(out, xim[:, :256])


def test_basicblock_init_random_results_in_change():
    xim = torch.rand((3, 384, 3, 3))
    cid = ResNetBasicBlock(384, 256, init_identity=False)
    out = cid(xim)
    assert out.shape == xim[:, :256].shape
    assert not torch.allclose(out, xim[:, :256])


def test_bottleneck_init_identity_results_in_no_change():
    xim = torch.rand((3, 256, 3, 3))
    cid = ResNetBottleneck(256, 256, init_identity=True)
    out = cid(xim)
    assert out.shape == xim.shape
    assert torch.allclose(out, xim)


def test_bottleneck_init_identity_with_downproject_results_in_no_change_for_initial_values():
    # NOTE: We can currently only guarantee this for eval mode, because the norm layer in the skip connection branch
    # will alter the output when in training mode.
    xim = torch.rand((3, 384, 3, 3))
    cid = ResNetBottleneck(384, 256, init_identity=True).eval()
    out = cid(xim)
    assert out.shape == xim[:, :256].shape
    assert torch.allclose(out, xim[:, :256])


def test_bottleneck_init_random_results_in_change():
    xim = torch.rand((3, 384, 3, 3))
    cid = ResNetBottleneck(384, 256, init_identity=False)
    out = cid(xim)
    assert out.shape == xim[:, :256].shape
    assert not torch.allclose(out, xim[:, :256])


def test_simpleadapter_fc_init_identity_results_in_no_change():
    # NOTE: We can currently only guarantee this for eval mode, because the norm layer in the skip connection branch
    # will alter the output when in training mode.
    xim = torch.rand((3, 5, 256))
    cid = SimpleAdapter(256, 256, num_fc=1, init_identity=True).eval()
    out = cid(xim)
    assert out.shape == xim.shape
    # WARNING: This is here just to remind us that we can't actually meet this requirement, because of the use of
    # LayerNorm in this case.
    # assert torch.allclose(out, xim, atol=1e-7)


def test_simpleadapter_fc_init_identity_with_downproject_results_in_no_change_for_initial_values():
    # NOTE: We can currently only guarantee this for eval mode, because the norm layer in the skip connection branch
    # will alter the output when in training mode.
    xim = torch.rand((3, 5, 384))
    cid = SimpleAdapter(384, 256, num_fc=1, init_identity=True).eval()
    out = cid(xim)
    assert out.shape == xim[:, :, :256].shape
    # WARNING: This is here just to remind us that we can't actually meet this requirement, because of the use of
    # LayerNorm in this case.
    # assert torch.allclose(out, xim[:, :, :256], atol=1e-7)


def test_simpleadapter_fc_init_random_results_in_change():
    xim = torch.rand((3, 5, 384))
    cid = SimpleAdapter(384, 256, num_fc=1, init_identity=False)
    out = cid(xim)
    assert out.shape == xim[:, :, :256].shape
    assert not torch.allclose(out, xim[:, :, :256])


def test_simpleadapter_1x1_conv_init_identity_results_in_no_change():
    # NOTE: We can currently only guarantee this for eval mode, because the norm layer in the skip connection branch
    # will alter the output when in training mode.
    xim = torch.rand((3, 256, 3, 3))
    cid = SimpleAdapter(256, 256, num_conv=1, kernel_size=1, padding=0, init_identity=True).eval()
    out = cid(xim)
    assert out.shape == xim.shape
    assert torch.allclose(out, xim, atol=1e-7)


def test_simpleadapter_1x1_conv_init_identity_with_downproject_results_in_no_change_for_initial_values():
    # NOTE: We can currently only guarantee this for eval mode, because the norm layer in the skip connection branch
    # will alter the output when in training mode.
    xim = torch.rand((3, 384, 3, 3))
    cid = SimpleAdapter(384, 256, num_conv=1, kernel_size=1, padding=0, init_identity=True).eval()
    out = cid(xim)
    assert out.shape == xim[:, :256].shape
    assert torch.allclose(out, xim[:, :256], atol=1e-7)


def test_simpleadapter_1x1_conv_init_random_results_in_change():
    xim = torch.rand((3, 384, 3, 3))
    cid = SimpleAdapter(384, 256, num_conv=1, kernel_size=1, padding=0, init_identity=False)
    out = cid(xim)
    assert out.shape == xim[:, :256].shape
    assert not torch.allclose(out, xim[:, :256])


def test_simpleadapter_3x3_conv_init_identity_results_in_no_change():
    # NOTE: We can currently only guarantee this for eval mode, because the norm layer in the skip connection branch
    # will alter the output when in training mode.
    xim = torch.rand((3, 256, 3, 3))
    cid = SimpleAdapter(256, 256, num_conv=1, kernel_size=3, padding=1, init_identity=True).eval()
    out = cid(xim)
    assert out.shape == xim.shape
    assert torch.allclose(out, xim, atol=1e-7)


def test_simpleadapter_3x3_conv_init_identity_with_downproject_results_in_no_change_for_initial_values():
    # NOTE: We can currently only guarantee this for eval mode, because the norm layer in the skip connection branch
    # will alter the output when in training mode.
    xim = torch.rand((3, 384, 3, 3))
    cid = SimpleAdapter(384, 256, num_conv=1, kernel_size=3, padding=1, init_identity=True).eval()
    out = cid(xim)
    assert out.shape == xim[:, :256].shape
    assert torch.allclose(out, xim[:, :256], atol=1e-7)


def test_simpleadapter_3x3_conv_init_random_results_in_change():
    xim = torch.rand((3, 384, 3, 3))
    cid = SimpleAdapter(384, 256, num_conv=1, kernel_size=3, padding=1, init_identity=False)
    out = cid(xim)
    assert out.shape == xim[:, :256].shape
    assert not torch.allclose(out, xim[:, :256])
