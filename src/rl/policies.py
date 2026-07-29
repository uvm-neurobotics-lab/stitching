"""
Turns a `policy` config into Stable-Baselines3 `policy_kwargs`.

The model config is split in two. `trunk` is the feature extractor this repo owns, stitches, and pretrains; it is
parsed by `assembly.model_from_config()` exactly as in supervised training. `policy` is the actor-critic head SB3
builds on top of the resulting feature vector, and is described by plain scalars rather than by parts.
"""
import torch
import torch.nn as nn

from assembly import ACTIVATIONS, _lookup
from rl.models import AssemblyExtractor
from utils import ensure_config_param, gt_zero, has_arg, of_type, one_of

# Config keys which hold a trunk, in priority order. "trunk" is the name to use; "model" and "assembly" are accepted
# so that a config written for `stitch_train.py` can be handed to `rl_train.py` unchanged.
TRUNK_KEYS = ("trunk", "model", "assembly")


def trunk_key(config):
    """ Which key holds this config's trunk. Raises if it is missing or ambiguous. """
    present = [k for k in TRUNK_KEYS if k in config]
    if not present:
        raise RuntimeError(f"No model found in the config. Supply one of: {', '.join(TRUNK_KEYS)}.")
    if len(present) > 1:
        raise RuntimeError(f"The config has more than one model: {present}. These are aliases, so supply only one "
                           "(prefer 'trunk').")
    return present[0]


def trunk_config(config):
    """
    Extract just the part of the config which describes the trunk, in the form `assembly.model_from_config()` wants.

    Deliberately not the whole config: this dict is stored in `policy_kwargs`, which SB3 pickles when saving a model.
    """
    key = trunk_key(config)
    if key == "assembly":
        # The legacy format, where the parts list, head, and reformat options are all separate top-level keys.
        extracted = {"assembly": config["assembly"]}
        for extra in ("head", "reformat_options"):
            if extra in config:
                extracted[extra] = config[extra]
        return extracted
    return {"model": config[key]}


def check_policy_config(config):
    """ Validate and fill in the `policy` portion of the config. Modifies `config` in place. """
    obs_mode = config["train_config"]["obs_mode"]

    ensure_config_param(config, "policy", of_type(dict), dflt={})
    ensure_config_param(config, ["policy", "features_dim"], gt_zero, dflt=64)
    ensure_config_param(config, ["policy", "net_arch"], of_type((list, dict)),
                        dflt={"pi": [64, 64], "vf": [64, 64]})
    ensure_config_param(config, ["policy", "activation_fn"], one_of(sorted(ACTIVATIONS)), dflt="tanh")
    ensure_config_param(config, ["policy", "ortho_init"], of_type(bool), dflt=True)
    ensure_config_param(config, ["policy", "normalize_images"], of_type(bool),
                        dflt=default_normalize_images(obs_mode))
    ensure_config_param(config, ["policy", "share_features_extractor"], of_type(bool), dflt=True)
    ensure_config_param(config, ["policy", "freeze_norm_stats"], of_type(bool), dflt=True)

    validate_trunk_part(config)


def validate_trunk_part(config):
    """
    Check that the top-level trunk part can accept the arguments we are going to hand it.

    `model_from_config()` passes `input_shape` and `num_classes` down to the top-level part. `Assembly` takes both;
    `Net` and `Subnet` forward their extra keyword arguments to the underlying model loader, where `num_classes`
    surfaces as a confusing TypeError from deep inside the architecture. Catch it here instead.
    """
    from assembly import part_class_from_name, validate_part

    key = trunk_key(config)
    if key == "assembly":
        return  # The legacy format always builds an Assembly.

    part_cfg = config[key]
    validate_part(part_cfg)
    cls_name = next(iter(part_cfg))
    part_class = part_class_from_name(cls_name)
    for arg in ("input_shape", "num_classes"):
        if not has_arg(part_class, arg):
            raise RuntimeError(
                f"The top-level model must accept '{arg}', because the observation shape and the feature width are "
                f"only known at runtime, but '{cls_name}' does not. Wrap it in an Assembly:\n"
                f"    {key}:\n      Assembly:\n        parts:\n        - {cls_name}: {{...}}\n"
                f"        head:\n          VectorHead: {{}}")


def default_normalize_images(obs_mode):
    """
    Whether SB3 should divide observations by 255.

    SB3 decides that a Box space is an image from its shape, dtype, and bounds alone, which makes MiniGrid's
    symbolic (7, 7, 3) uint8 observations look exactly like pixels. They are not: they are (OBJECT_IDX, COLOR_IDX,
    STATE) codes with a maximum value around 10, so dividing by 255 would squash them into a sliver of [0, 1].
    """
    return obs_mode in ("rgb", "rgb_full")


def policy_name_for(config):
    """ SB3's policy alias for this observation shape. We always supply our own extractor, so this only picks the
    defaults we are about to override anyway; choosing by shape keeps SB3 from warning about a mismatch. """
    return "CnnPolicy" if config["train_config"]["obs_mode"] != "flat" else "MlpPolicy"


def policy_kwargs_from_config(config):
    """ Build the `policy_kwargs` dict to hand to the algorithm's constructor. """
    policy_cfg = config["policy"]
    train_cfg = config["train_config"]

    # SB3 constructs the optimizer itself, passing the learning rate positionally from its own schedule, so passing
    # `lr` here too would be a duplicate argument. The learning rate reaches the algorithm via `learning_rate`.
    optimizer_kwargs = {k: v for k, v in train_cfg["optimizer_args"].items() if k != "lr"}

    return {
        "features_extractor_class": AssemblyExtractor,
        "features_extractor_kwargs": {
            "trunk_config": trunk_config(config),
            "features_dim": policy_cfg["features_dim"],
            "freeze_norm_stats": policy_cfg["freeze_norm_stats"],
        },
        "net_arch": policy_cfg["net_arch"],
        "activation_fn": _lookup(ACTIVATIONS, policy_cfg["activation_fn"], "activation"),
        "ortho_init": policy_cfg["ortho_init"],
        "normalize_images": policy_cfg["normalize_images"],
        "share_features_extractor": policy_cfg["share_features_extractor"],
        "optimizer_class": getattr(torch.optim, train_cfg["optimizer"]),
        "optimizer_kwargs": optimizer_kwargs,
    }
