"""
A class for assembling blocks and adapters into a single module.
"""
import inspect
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

import adapters
import utils
from utils.models import load_model, load_subnet


def img2bhwc(x, **kwargs):
    return x.permute(0, 2, 3, 1)


def bhwc2img(x, **kwargs):
    return x.permute(0, 3, 1, 2)


def img2token(x, **kwargs):
    # FIXME: Add position encoding??
    return x.flatten(2).transpose(1, 2)


def token2img(x, **kwargs):
    img_sz = int(np.sqrt(x.shape[1]))
    if x.shape[1] != img_sz ** 2:
        raise RuntimeError(f"Sequence length ({x.shape[1]}) must be a perfect square to transform to image format.")
    return x.view(-1, img_sz, img_sz, x.shape[2]).permute(0, 3, 1, 2)


def nonstrict_token2img(x, **kwargs):
    """WARNING: Deprecated."""
    # Drop the extra tokens if there are any. We expect the number of tokens to be a perfect square, so we can map it
    # onto a square image format. But ViT, for instance, has an extra token.
    img_sz = int(np.sqrt(x.shape[1]))
    token_limit = img_sz ** 2
    x = x[:, :token_limit, :]
    return x.view(-1, img_sz, img_sz, x.shape[2]).permute(0, 3, 1, 2)


def bert2img(x, **kwargs):
    """
    Transform from a BERT-style sequence representation: the leading token is a special slot reserved for output, and
    the rest of the sequence is the image patches (in row-major order).
    """
    img_sz = int(np.sqrt(x.shape[1]))
    if x.shape[1] != (img_sz ** 2 + 1):
        raise RuntimeError(f"Sequence length should be a perfect square plus one output token: {x.shape[1]}.")
    out_token = x[:, 0, :]
    x = x[:, 1:, :]
    images = x.view(-1, img_sz, img_sz, x.shape[2]).permute(0, 3, 1, 2)
    # Now broadcast the class token over (H x W) to create a special "channel".
    # Reshape to [B, C, 1, 1] and broadcast to [B, C, H, W].
    out_img = out_token[:, :, None, None].expand(-1, -1, img_sz, img_sz)
    # Concatenate along the channel dimension
    return torch.cat([images, out_img], dim=1)  # [B, 2C, H, W]


def img2bert(x, **kwargs):
    """
    Transform to a BERT-style sequence representation: the leading token is a special slot reserved for output, and
    the rest of the sequence is the image patches (in row-major order).
    FIXME: Add position encoding??
    FIXME: To be able to learn what to put as the output token, we would need the adapter to know that it should output
           an extra pixel. Without that, the only thing we can do is add a constant for output token. Or perform some
           fixed operation, like averaging all the patch values.
    """
    x = x.flatten(2)

    class_token_action = kwargs.get("class_token")
    # Add class/output token to beginning of sequence.
    if (not class_token_action) or class_token_action == "zeros" or class_token_action == "zeroes":
        # This one simply adds a token consisting of all 0's.
        x = torch.cat([torch.zeros_like(x[:, :, 0:1]), x], dim=2)
    elif class_token_action == "ones":
        # This one simply adds a token consisting of all 1's.
        x = torch.cat([torch.ones_like(x[:, :, 0:1]), x], dim=2)
    elif class_token_action == "avg" or class_token_action == "mean":
        # This one adds a token which is the average of all the other tokens.
        x = torch.cat([torch.mean(x, dim=2, keepdim=True), x], dim=2)

    return x.transpose(1, 2)


def to_square_img_size(size_1d):
    img_sz = int(np.sqrt(size_1d))
    if size_1d != img_sz ** 2:
        raise RuntimeError(f"Sequence length ({size_1d}) must be a perfect square to transform to image format.")
    size_1d = (img_sz, img_sz)
    return size_1d


TRANSFORM = {
    ("img", "bhwc"): img2bhwc,
    ("bhwc", "img"): bhwc2img,
    ("img", "bert"): img2bert,
    ("bert", "img"): bert2img,
    ("img", "token"): img2token,
    ("token", "img"): token2img,
    ("bhwc", "bert"): lambda x, **kwargs: img2bert(bhwc2img(x, **kwargs), **kwargs),
    ("bert", "bhwc"): lambda x, **kwargs: img2bhwc(bert2img(x, **kwargs), **kwargs),
    ("bhwc", "token"): lambda x, **kwargs: img2token(bhwc2img(x, **kwargs), **kwargs),
    ("token", "bhwc"): lambda x, **kwargs: img2bhwc(token2img(x, **kwargs), **kwargs),
}


def parse_format_from_tensor(x, cur_fmt):
    if not isinstance(cur_fmt, str):
        cur_fmt, _ = cur_fmt  # Current size is ignored, b/c we just use the size of the tensor.

    if cur_fmt == "img":
        cur_ch = x.shape[1]
        cur_sz = tuple(x.shape[2:])
    elif cur_fmt == "bhwc":
        cur_ch = x.shape[3]
        cur_sz = tuple(x.shape[1:3])
    elif cur_fmt == "bert":
        cur_ch = x.shape[2]
        cur_sz = to_square_img_size(x.shape[1] - 1)
    elif cur_fmt == "token":
        cur_ch = x.shape[2]
        cur_sz = to_square_img_size(x.shape[1])
    else:
        raise RuntimeError(f"Unrecognized format type: '{cur_fmt}'")

    return cur_fmt, cur_ch, cur_sz


def reformat(x: torch.Tensor, cur_fmt: Union[str, tuple, list], target_fmt: Optional[Union[str, tuple, list]],
             reformat_options: Optional[dict] = None):
    """
    Converts the given tensor from current format into desired format.

    Format descriptions are a pair of [string, size]. String should be a type available in `TRANSFORM` and describes the
    format of the input: 2D image, 1D sequence, etc. Size is the size of the image or sequence. Sizes are in either
    [H, W] or [C, H, W] format for images, or [C, L] or [L] format for sequences. All inputs are assumed to represent
    images. Therefore, we always convert to an image before doing any resizing; resizing behaves as image
    upsampling/downsampling.

    Args:
        x: The tensor to transform.
        cur_fmt: The current format.
        target_fmt: The desired format.
        reformat_options: Options to be passed along as kwargs to the transform function.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    if not target_fmt:
        # If no target format is requested, then we can just keep the same format.
        return x

    # Parse inputs.
    cur_fmt, cur_ch, cur_sz = parse_format_from_tensor(x, cur_fmt)

    target_ch = None
    target_sz = None
    if not isinstance(target_fmt, str):
        target_fmt, target_sz = target_fmt
        if isinstance(target_sz, int):
            # If a 1D size is requested, determine the equivalent square image size.
            target_sz = to_square_img_size(target_sz)
        elif isinstance(target_sz, (tuple, list)) and len(target_sz) == 2:
            if target_fmt == "img" or target_fmt == "bhwc":
                # If format is image-like, then assume this is [H, W].
                target_sz = tuple(int(s) for s in target_sz)
            else:
                # If format is sequence-like, then assume this is [C, H*W]. Extract the C and get a 2D size.
                target_ch = target_sz[0]
                target_sz = to_square_img_size(target_sz[1])
        elif isinstance(target_sz, (tuple, list)) and len(target_sz) == 3:
            # If we have 3 values, they must be in [C, H, W] format. Split up the channel and spatial values.
            target_ch = target_sz[0]
            target_sz = tuple(int(s) for s in target_sz[1:])
        else:
            raise RuntimeError(f"Target size should be either an int or a pair but got: {target_sz}")

    reformat_options = {} if reformat_options is None else reformat_options

    # Ensure number of channels is expected.
    if cur_ch and target_ch and (cur_ch != target_ch):
        error = True
        # Exception: if going from BERT to image, we will double the number of channels due to the special class token.
        if cur_fmt == "bert" and target_fmt == "img" and ((2 * cur_ch) == target_ch):
            error = False
        if error:
            raise RuntimeError(f"Current channels ({cur_ch}) and target channels ({target_ch}) do not match.")

    # Convert the size.
    # FIXME: If we do a resize from BERT to BERT, we lose the extra class info, even though it would be easy to retain.
    #        We should have a way to keep it.
    if target_sz is not None and cur_sz != target_sz:
        if cur_fmt != "img":
            x = TRANSFORM[(cur_fmt, "img")](x, **reformat_options)
            cur_fmt = "img"
        x = nn.functional.interpolate(x, target_sz, mode="bilinear")

    # Convert the format.
    if cur_fmt != target_fmt:
        x = TRANSFORM[(cur_fmt, target_fmt)](x, **reformat_options)

    return x


def get_in_fmt(part, dflt=None):
    ret = getattr(part, "in_fmt", dflt)
    return dflt if ret is None else ret


def get_out_fmt(part, dflt=None):
    ret = getattr(part, "out_fmt", dflt)
    return dflt if ret is None else ret


def part_forward(part, x, cur_fmt, reformat_options=None, part_index=None):
    """
    Run the forward pass of the given part, using the given input `x`. This function enables tracking of the format of
    input `x`, and automatic reformatting to conform to the constraints of the `part`. It returns both the updated `x`
    and the updated format object.

    Args:
        part: The module to run.
        x: The input to the module.
        cur_fmt: The current known format of the input.
        reformat_options: Options to be forwarded to `reformat()`.
        part_index: The index of the given part, if applicable.

    Returns:
        torch.Tensor: The new output.
        tuple: New format description corresponding to the output tensor.
    """
    try:  # Main function body.

        x = reformat(x, cur_fmt, get_in_fmt(part), reformat_options)  # Convert to the format requested by part.
        # The part may have support for built-in format-tracking. (E.g., if we have nested Assemblies.) In which
        # case, we do not need to update the format manually.
        if "cur_fmt" in inspect.signature(part.forward).parameters:
            x, cur_fmt = part(x, cur_fmt)  # Execute the part.
        else:
            x = part(x)  # Execute the part.
            cur_fmt = get_out_fmt(part, cur_fmt)  # Update the format; if None, then format is unchanged.
        if isinstance(x, dict):  # If part is a GraphModule, we need to extract its output from a dict.
            x = list(x.values())[0]
        return x, cur_fmt

    except Exception as e:  # Wrapper just to inform the user where the error occurred, if applicable.
        if part_index is None:
            raise
        else:
            raise RuntimeError(f"Error in the part at index {part_index}. This can be a fault in the part itself, or "
                               f"in reformatting just prior to this part. Error:\n{str(e)}")


class Net(nn.Module):

    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.in_fmt = "img"
        self.out_fmt = "vector"
        self.net = load_model(model_name=model_name, **kwargs)

    def forward(self, x):
        return self.net(x)


class Subnet(nn.Module):

    def __init__(self, model_name, in_format=None, out_format=None, **kwargs):
        super().__init__()
        self.in_fmt = in_format
        self.out_fmt = out_format
        self.net = load_subnet(model_name=model_name, **kwargs)

    def forward(self, x):
        return self.net(x)


class BaseBlock(nn.Module):
    """Used as the base of assemblies in the DeRy paper."""

    def __init__(self, input_shape, channels):
        super().__init__()
        self.conv = nn.Conv2d(input_shape[0], channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_fmt = "img"
        self.out_fmt = "img"

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ClassifierHead(nn.Module):

    def __init__(self, input_shape, num_classes):
        super().__init__()
        if len(input_shape) != 1:
            raise RuntimeError(f"Cannot stack {type(self).__name__} on top of output of shape: {input_shape}.")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(input_shape[0], num_classes)
        self.in_fmt = "img"
        self.out_fmt = "vector"

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return self.linear(x)


def validate_part_list(part_list):
    if not part_list:
        raise ValueError(f"No assembly parts specified. You must specify at least one part.")
    if not isinstance(part_list, list):
        raise ValueError(f"Assembly config should be a list of parts.")
    return all(validate_part(c) for c in part_list)


def validate_part(part_cfg):
    if not isinstance(part_cfg, dict):
        raise ValueError(f"Unrecognized part config format: {part_cfg}")
    if len(part_cfg) > 1:
        raise ValueError(f"Each part in the part list should consist of a single class with args.")
    cls_name, args = next(iter(part_cfg.items()))
    part_class_from_name(cls_name)
    if args and not isinstance(args, dict):
        raise ValueError(f"Expected a dictionary of arguments for {cls_name}, but instead got: {args}")
    return True


def part_class_from_name(cls_name):
    if hasattr(adapters, cls_name):
        return getattr(adapters, cls_name)
    elif cls_name in globals():
        return globals()[cls_name]
    else:
        raise RuntimeError(f"Part class not found: '{cls_name}'")


def part_from_config(part_cfg):
    if not validate_part(part_cfg):
        raise ValueError(f"Invalid part config:\n{part_cfg}")
    cls_name, args = next(iter(part_cfg.items()))
    PartClass = part_class_from_name(cls_name)
    args = args.copy() if args is not None else {}
    if "frozen" in args:
        del args["frozen"]
    return PartClass(**args)


def model_from_config(config, input_shape=None):
    """
    Instantiate a network from a config specifying one or more "parts".

    Args:
        config: The config where we can find a "model" or "assembly" subconfig.
        input_shape: (Optional) The size of the input tensor that the model can expect, different for different
            datasets. If using the "model" config format, this must be specified in config manually (most assembly
            configs are not flexible enough to allow different sizes anyhow).
    Returns:
        nn.Module: A callable model.
    """
    if "assembly" in config and "model" in config:
        raise RuntimeError('You may only specify one of "assembly" or "model".')
    elif "assembly" in config:
        # Slightly deprecated; this parses an older config format.
        return Assembly(config["assembly"], config.get("head"), input_shape=input_shape,
                        reformat_options=config.get("reformat_options"))
    elif "model" in config:
        # "model" is the new config item, preferred over "assembly". This allows us to use any type of part at the top
        # level, not just Assembly.
        return part_from_config(config["model"])
    else:
        raise RuntimeError(f"Model config not found. Config needs either 'model' or 'assembly' keys.")


def freeze(part):
    for param in part.parameters():
        param.requires_grad = False


class Assembly(nn.Module):
    """
    A class which takes a list of blocks and a list of adapters and assembles them all in a sequence.

    The user can choose to leave out the adapters if they only want to use blocks (this assumes all blocks'
    inputs/outputs are compatible with each other). Also provides an option for adding a prescribed block at the end
    of the model.

    This is partly inspired by Deep Model Reassembly:
    https://github.com/Adamdad/DeRy/blob/main/mmcls_addon/models/backbones/dery.py
    """

    def __init__(
            self,
            parts,
            head=None,
            input_shape=None,
            reformat_options=None,  # FIXME: finish implementing BERT transform options
    ):
        super().__init__()
        self.reformat_options = reformat_options

        if not validate_part_list(parts):
            raise ValueError(f"Invalid part list config:\n{parts}")

        part_list = []
        for c in parts:
            part = part_from_config(c)
            part_list.append(part)
            part_args = next(iter(c.values()))
            if part_args is not None and part_args.get("frozen"):
                freeze(part)
        self.parts = nn.ModuleList(part_list)

        if head:
            if not input_shape:
                raise ValueError("To use a classifier head, you must specify the expected input shape.")
            if not isinstance(head, dict) or len(head) > 1:
                raise ValueError(f"Unrecognized head config format: {head}."
                                 " Should consist of a single class with args.")
            output_shape = utils.calculate_output_shape(self.trunk_forward, input_shape)
            head_args = next(iter(head.values()))
            head_args["input_shape"] = output_shape
            self.head = part_from_config(head)
            if head_args.get("frozen"):
                freeze(self.head)
        else:
            self.head = None

        self.in_fmt = get_in_fmt(self.parts[0])
        self.out_fmt = get_out_fmt(self.head) if self.head else get_out_fmt(self.parts[0])

    def trunk_forward(self, x, cur_fmt=None):
        if cur_fmt is None:
            cur_fmt = "img"  # Assume this is the very first input, therefore it is an image.
        for i, part in enumerate(self.parts):
            x, cur_fmt = part_forward(part, x, cur_fmt, self.reformat_options, i)
        return x, cur_fmt

    def forward(self, x, cur_fmt=None):
        # If `cur_fmt` is passed to us, then we should also return it. Otherwise, act like we have one input/output.
        fmt_desired = (cur_fmt is not None)
        x, cur_fmt = self.trunk_forward(x, cur_fmt)
        if self.head:
            x, cur_fmt = part_forward(self.head, x, cur_fmt, self.reformat_options)
        return (x, cur_fmt) if fmt_desired else x


class ParallelPart(nn.Module):
    """
    A class which takes a list of blocks and assembles them in parallel (rather than in series, like the Assembly).

    This can be used as a sub-part of an Assembly, and can also have Assemblies as sub-parts of itself.
    """

    def __init__(
            self,
            parts,
            agg,
            in_format=None,
            out_format=None,
            reformat_options=None,
    ):
        super().__init__()
        self.agg = agg
        self.in_fmt = in_format
        self.out_fmt = out_format
        if len(out_format) != 2 or len(out_format[1]) != 3:
            raise RuntimeError("Output format must be fully specified for ParallelAdapter. "
                               f"Instead we got: {out_format}")
        self.reformat_options = reformat_options

        if not validate_part_list(parts):
            raise ValueError(f"Invalid part list config:\n{parts}")

        part_list = []
        for c in parts:
            part = part_from_config(c)
            part_list.append(part)
            part_args = next(iter(c.values()))
            if part_args is not None and part_args.get("frozen"):
                freeze(part)
        self.parts = nn.ModuleList(part_list)

    def forward(self, x, cur_fmt=None):
        if cur_fmt is None:
            cur_fmt = "img"  # Assume this is the very first input, therefore it is an image.

        # Run all the parallel parts.
        # NOTE: In theory this loop could be parallelized, in the future. Only useful if model parallelism is used.
        results = []
        for i, part in enumerate(self.parts):
            results.append(list(part_forward(part, x, cur_fmt, self.reformat_options, i)))

        # Now aggregate the parts together.
        if self.agg == "avg":

            # Convert all results to the final expected output format.
            for i, res in enumerate(results):
                res[0] = reformat(res[0], res[1], self.out_fmt, self.reformat_options)

            # Now combine them by mean.
            x = torch.stack([r[0] for r in results]).mean(dim=0)

        elif self.agg == "concat":

            # Convert formats and ensure number of channels adds up to the expected amount.
            total_channels = 0
            for i, res in enumerate(results):
                # Count the number of channels in each output.
                _, num_channels, _ = parse_format_from_tensor(*res)
                total_channels += num_channels
                # Ensure that everything other than the channels are converted to the expected format.
                target_fmt = [self.out_fmt[0], self.out_fmt[1][1:]]  # Drop channels from target format.
                res[0] = reformat(res[0], res[1], target_fmt, self.reformat_options)
            if total_channels != self.out_fmt[1][0]:
                msg = (f"Expected the total number of channels to be {self.out_fmt[1][0]}, but instead got "
                       f"{total_channels}. Results:")
                for res in results:
                    msg += f"\nShape={res[0].shape}, Format={res[1]}"
                raise RuntimeError(msg)

            # Now we can stack the outputs on the channel dimension.
            x = torch.concat([r[0] for r in results], dim=1)

        else:
            raise RuntimeError(f"Unrecognized aggregation function: {self.agg}")

        return x, self.out_fmt
