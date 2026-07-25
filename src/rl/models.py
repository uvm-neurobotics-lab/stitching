"""
The bridge between this repo's models and Stable-Baselines3.

SB3 splits a policy into a "features extractor" and the actor/critic heads built on top of it. That split is exactly
the seam we need: the extractor is the trunk, built by `assembly.model_from_config()` just as it is for supervised
training, so a trunk can be new, pretrained, or a stitching of several models without the RL code knowing.
"""
import logging

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.batchnorm import _NormBase

from assembly import model_from_config
from utils.logging import eval_mode

# The attributes under which SB3 may store a features extractor. Which of these exist, and whether they are the same
# object, depends on `share_features_extractor`.
EXTRACTOR_ATTRS = ("features_extractor", "pi_features_extractor", "vf_features_extractor")


class AssemblyExtractor(BaseFeaturesExtractor):
    """
    Presents a model built by `assembly.model_from_config()` as an SB3 features extractor.

    Two things are worth knowing about what SB3 hands us:
      - `observation_space` has already been through `VecTransposeImage`, so image shapes are [C, H, W] -- which is
        this repo's "img" format, and the format `Assembly.trunk_forward()` assumes for its input. Deriving the
        input shape from this space rather than from the raw environment is what keeps the two conventions aligned.
      - `forward()` receives a float tensor which has already been through SB3's `preprocess_obs`, including the
        division by 255 when `normalize_images` is on.
    """

    def __init__(self, observation_space, trunk_config, features_dim=None, freeze_norm_stats=True):
        # BaseFeaturesExtractor demands a positive width up front, but we cannot know the real one until the trunk
        # has been built and run. Pass a placeholder and correct it below, the same way SB3's own NatureCNN does.
        super().__init__(observation_space, features_dim=1)

        input_shape = tuple(observation_space.shape)
        self.model = model_from_config(trunk_config, input_shape=input_shape, num_classes=features_dim)

        out_dim = self._dry_run(input_shape)
        if features_dim is not None and out_dim != features_dim:
            raise RuntimeError(f"The configured trunk produces {out_dim} features, but policy.features_dim is "
                               f"{features_dim}. Either set policy.features_dim to {out_dim}, or give the trunk a "
                               "head whose width follows num_classes (FeatureHead does).")
        self._features_dim = out_dim

        self.freeze_norm_stats = freeze_norm_stats
        self._apply_norm_freeze()

        # Snapshot the weights so they can be restored after SB3 re-initializes the policy. See
        # `restore_pretrained_weights()` for why that is necessary.
        self._weight_snapshot = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _dry_run(self, input_shape):
        """ Run one batch through the trunk to discover how wide its output is. """
        with eval_mode(self.model):
            out = self.model(torch.zeros((2,) + input_shape))
        if out.ndim != 2:
            raise RuntimeError(f"The trunk must produce a flat feature vector of shape [batch, features], but it "
                               f"produced {tuple(out.shape)}. Add a head which flattens its output; FeatureHead "
                               "does this.")
        return out.shape[1]

    def _apply_norm_freeze(self):
        """
        Hold frozen normalization layers in eval mode.

        A frozen part is already handled by `Assembly.train()`, but a normalization layer whose parameters were
        frozen individually is not. Without this, PPO's update passes -- which run the policy in training mode --
        would keep drifting its running statistics even though none of its weights can change.
        """
        if not self.freeze_norm_stats:
            return
        for module in self.model.modules():
            if isinstance(module, _NormBase) and module.track_running_stats:
                params = list(module.parameters(recurse=False))
                if params and not any(p.requires_grad for p in params):
                    module.eval()

    def train(self, mode: bool = True):
        # SB3 flips the whole policy in and out of training mode around every update, so re-apply after each flip.
        super().train(mode)
        self._apply_norm_freeze()
        return self

    def forward(self, observations):
        return self.model(observations)

    def restore_pretrained(self):
        """ Restore the weights the trunk was constructed with, discarding any re-initialization since. """
        self.model.load_state_dict(self._weight_snapshot)
        self._apply_norm_freeze()

    def snapshot_weights(self):
        """ Take the current weights as the ones `restore_pretrained()` should restore. """
        self._weight_snapshot = {k: v.detach().clone() for k, v in self.model.state_dict().items()}


def iter_extractors(sb3_model):
    """ Yield each distinct `AssemblyExtractor` in the model's policy, without visiting a shared one twice. """
    seen = set()
    for attr in EXTRACTOR_ATTRS:
        extractor = getattr(sb3_model.policy, attr, None)
        if isinstance(extractor, AssemblyExtractor) and id(extractor) not in seen:
            seen.add(id(extractor))
            yield extractor


def restore_pretrained_weights(sb3_model):
    """
    Undo SB3's re-initialization of the features extractor.

    `ActorCriticPolicy._build()` finishes by applying orthogonal initialization to the features extractor, the mlp
    extractor, and the action and value heads. That is the right thing for the heads -- the small gain on the action
    head keeps the initial policy close to uniform, which stabilizes early training -- but applied to the features
    extractor it silently overwrites every Conv2d and Linear weight in a pretrained trunk. The resulting run looks
    entirely plausible and has thrown away the pretraining.

    Turning `ortho_init` off would avoid it at the cost of the heads' initialization, so instead we put the trunk's
    weights back afterwards and leave the heads as SB3 initialized them. Must be called after the policy is built.

    Returns:
        int: How many extractors were restored.
    """
    count = 0
    for extractor in iter_extractors(sb3_model):
        extractor.restore_pretrained()
        count += 1
    return count


def load_trunk_weights(sb3_model, ckp_path, strict=True):
    """
    Load trunk weights from a checkpoint, for `--load-from`.

    Accepts either a bare state dict or a checkpoint dict with a "model" key, which is the format both the supervised
    trainer and this one write. That is what allows a trunk trained by `stitch_train.py` to be picked up here.
    """
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    count = 0
    for extractor in iter_extractors(sb3_model):
        missing, unexpected = extractor.model.load_state_dict(state_dict, strict=strict)
        if missing:
            logging.warning(f"Missing keys when loading trunk weights: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys when loading trunk weights: {unexpected}")
        # These are now the weights to keep, so a later restore must not revert to the constructed ones.
        extractor.snapshot_weights()
        count += 1
    if not count:
        raise RuntimeError(f"Could not load weights from {ckp_path}: the policy has no AssemblyExtractor.")
    logging.info(f"Loaded trunk weights from: {ckp_path}")
    return count


def trunk_state_dict(sb3_model):
    """ The trunk's state dict, in the same layout `stitch_train.py` writes, so checkpoints interoperate. """
    for extractor in iter_extractors(sb3_model):
        return extractor.model.state_dict()
    return {}


def apply_freezing(sb3_model, config):
    """ Honor the `unfrozen` config flag, which overrides any `frozen` marks in the model config. """
    if config.get("unfrozen"):
        from assembly import unfreeze
        for extractor in iter_extractors(sb3_model):
            unfreeze(extractor.model)


def describe_parameters(sb3_model):
    """ A human-readable summary of how the policy's parameters are split between the trunk and the heads. """
    from utils import num_params, num_trainable_params

    policy = sb3_model.policy
    total, trainable = num_params(policy), num_trainable_params(policy)
    trunk_total = sum(num_params(e.model) for e in iter_extractors(sb3_model))
    return (f"Model has {total:.3e} total and {trainable:.3e} trainable params "
            f"({trunk_total:.3e} in the trunk, {total - trunk_total:.3e} in the policy heads).")


def named_norm_layers(module):
    """ Yield (name, module) for every normalization layer which tracks running statistics. Useful in tests. """
    for name, child in module.named_modules():
        if isinstance(child, (_NormBase, nn.LayerNorm, nn.GroupNorm)):
            yield name, child
