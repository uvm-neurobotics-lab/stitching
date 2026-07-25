from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl.algo import build_model, model_from_config
from rl.envs import make_vec_envs
from rl.models import AssemblyExtractor, restore_pretrained_weights
from tests.rl.configs import CONV_TRUNK, smoke_config, validated

SYMBOLIC_SPACE = gym.spaces.Box(low=0, high=255, shape=(3, 7, 7), dtype=np.uint8)


def make_extractor(features_dim=64, trunk=None, **kwargs):
    return AssemblyExtractor(SYMBOLIC_SPACE, trunk or {"model": deepcopy(CONV_TRUNK)},
                             features_dim=features_dim, **kwargs)


def frozen_trunk():
    """ A trunk whose convolutional body is frozen, standing in for a pretrained backbone. """
    trunk = deepcopy(CONV_TRUNK)
    trunk["Assembly"]["parts"][0]["Net"]["norm_type"] = "bn"  # Give it running statistics to protect.
    trunk["Assembly"]["parts"][0]["Net"]["frozen"] = True
    return {"model": trunk}


def test_extractor_produces_the_requested_width():
    extractor = make_extractor(features_dim=64)
    assert extractor.features_dim == 64
    assert extractor(torch.zeros(2, 3, 7, 7)).shape == (2, 64)


def test_extractor_reports_a_mismatched_width():
    # An Assembly does not pass num_classes down to its parts, only to its head, so a part with an explicit width
    # and no head pins the output width regardless of what the policy asked for. Say so, rather than letting the
    # shape mismatch surface later from inside the policy.
    pinned = {"model": {"Assembly": {"parts": [{"MLP": {"in_format": None, "in_features": 147,
                                                        "out_features": 11}}]}}}
    with pytest.raises(RuntimeError, match="produces 11 features, but policy.features_dim is 64"):
        AssemblyExtractor(SYMBOLIC_SPACE, pinned, features_dim=64)


def test_extractor_requires_a_flat_trunk_output():
    # A bare convolutional trunk emits a feature map; it needs a head to flatten it.
    no_head = {"model": {"Assembly": {"parts": deepcopy(CONV_TRUNK["Assembly"]["parts"])}}}
    with pytest.raises(RuntimeError, match="flat feature vector"):
        AssemblyExtractor(SYMBOLIC_SPACE, no_head, features_dim=64)


def test_extractor_handles_non_contiguous_observations():
    # SB3's VecTransposeImage hands us a channels-last array viewed as [B, C, H, W].
    extractor = make_extractor()
    obs = torch.zeros(2, 7, 7, 3).permute(0, 3, 1, 2)
    assert not obs.is_contiguous()
    assert extractor(obs).shape == (2, 64)


def test_frozen_norm_layers_stay_in_eval_mode():
    extractor = make_extractor(trunk=frozen_trunk())
    extractor.train(True)  # This is what SB3 does around every update.

    norms = [m for m in extractor.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms, "Test setup should have produced some BatchNorm layers."
    assert not any(m.training for m in norms), "Frozen normalization layers must not follow the model into training."

    before = [m.running_mean.clone() for m in norms]
    extractor(torch.randn(4, 3, 7, 7))
    assert all(torch.equal(m.running_mean, b) for m, b in zip(norms, before)), \
        "A frozen trunk's running statistics drifted during a forward pass."


def norm_trunk():
    """ An unfrozen trunk which has normalization layers with running statistics. """
    trunk = deepcopy(CONV_TRUNK)
    trunk["Assembly"]["parts"][0]["Net"]["norm_type"] = "bn"
    return {"model": trunk}


def test_norm_freeze_covers_individually_frozen_layers(freeze=True):
    # A part marked `frozen` is already handled by Assembly.train(). This flag covers the other case: a layer whose
    # parameters were frozen on their own, inside a part which is otherwise trainable.
    extractor = make_extractor(trunk=norm_trunk(), freeze_norm_stats=freeze)
    norms = [m for m in extractor.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms, "Test setup should have produced some BatchNorm layers."
    for param in norms[0].parameters():
        param.requires_grad = False

    extractor.train(True)
    assert norms[0].training is (not freeze), "freeze_norm_stats should decide whether the layer follows train()."
    assert norms[1].training, "Layers whose parameters are still trainable must follow the model's mode."


def test_norm_freeze_can_be_disabled():
    test_norm_freeze_covers_individually_frozen_layers(freeze=False)


def test_pretrained_weights_survive_policy_construction():
    # This is the one that matters. ActorCriticPolicy._build() finishes by orthogonally re-initializing every
    # Conv2d and Linear it can reach, including the ones inside our features extractor. Without restoring them, a
    # run with a pretrained trunk looks completely normal and has silently discarded the pretraining.
    config = validated()
    train_env = make_vec_envs(config)
    try:
        sb3_model = model_from_config(config, train_env, "cpu")
        extractor = sb3_model.policy.features_extractor
        # The snapshot is keyed by the trunk's own state dict, without the extractor's "model." prefix.
        snapshot = extractor._weight_snapshot
        first_conv = "parts.0.net.encoder.block1.conv0.weight"

        assert not torch.equal(extractor.model.state_dict()[first_conv], snapshot[first_conv]), \
            "Expected SB3 to have re-initialized the trunk; if it no longer does, this guard is obsolete."

        assert restore_pretrained_weights(sb3_model) == 1
        assert torch.equal(extractor.model.state_dict()[first_conv], snapshot[first_conv]), \
            "The trunk's constructed weights were not restored."
    finally:
        train_env.close()


def test_policy_heads_keep_their_own_initialization():
    # We restore the trunk but deliberately leave SB3's initialization of the heads alone: the small gain on the
    # action head keeps the initial policy near-uniform, which is a real stabilizer.
    config = validated()
    train_env = make_vec_envs(config)
    try:
        sb3_model = model_from_config(config, train_env, "cpu")
        before = sb3_model.policy.action_net.weight.clone()
        restore_pretrained_weights(sb3_model)
        assert torch.equal(sb3_model.policy.action_net.weight, before)
        # SB3 applies a gain of 0.01 to the action head, so its weights should be small.
        assert sb3_model.policy.action_net.weight.abs().max() < 0.5
    finally:
        train_env.close()


def test_load_from_accepts_a_supervised_checkpoint(tmp_path):
    # A trunk trained by stitch_train.py must be loadable here; that interoperability is the point of the repo.
    from rl.models import load_trunk_weights

    config = validated()
    train_env = make_vec_envs(config)
    try:
        donor = build_model(config, train_env, "cpu")
        donor_trunk = donor.policy.features_extractor.model
        with torch.no_grad():  # Make the donor's weights distinctive.
            for param in donor_trunk.parameters():
                param.fill_(0.5)
        # Exactly the layout utils/logging.py writes for a supervised run.
        ckpt_path = tmp_path / "checkpoint.pth"
        torch.save({"model": donor_trunk.state_dict(), "epoch": 1}, ckpt_path)

        receiver = build_model(config, train_env, "cpu")
        load_trunk_weights(receiver, ckpt_path, strict=True)
        for param in receiver.policy.features_extractor.model.parameters():
            assert torch.allclose(param, torch.full_like(param, 0.5))

        # A later restore must keep the loaded weights, not revert to the constructed ones.
        restore_pretrained_weights(receiver)
        for param in receiver.policy.features_extractor.model.parameters():
            assert torch.allclose(param, torch.full_like(param, 0.5))
    finally:
        train_env.close()


def test_unfrozen_flag_overrides_frozen_parts():
    config = validated(unfrozen=True, trunk=deepcopy(frozen_trunk()["model"]))
    train_env = make_vec_envs(config)
    try:
        sb3_model = build_model(config, train_env, "cpu")
        trunk = sb3_model.policy.features_extractor.model
        assert all(p.requires_grad for p in trunk.parameters()), "--unfrozen should make everything trainable."
    finally:
        train_env.close()
