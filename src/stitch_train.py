"""
A script to train an architecture and store the training trajectory in the corresponding cache.

To test this script, try:
    python stitch_train.py -c tests/stitch-resnet-18.yml --st
"""

import logging
import sys
from pathlib import Path

import utils.argparsing as argutils
import utils.datasets as datasets


def create_arg_parser(desc, allow_abbrev=True, allow_id=True):
    """
    Creates the argument parser for this program.

    Args:
        desc (str): The human-readable description for the arg parser.
        allow_abbrev (bool): The `allow_abbrev` argument to `argparse.ArgumentParser()`.
        allow_id (bool): The `allow_id` argument to the `argutils.add_wandb_args()` function.

    Returns:
        argutils.ArgParser: The parser.
    """
    parser = argutils.create_parser(desc, allow_abbrev=allow_abbrev)
    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Training config file.")
    datasets.add_dataset_arg(parser, dflt_data_dir=Path(__file__).parent / "data")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save the model weights at the end of each epoch.")
    parser.add_argument("--no-eval-checkpoints", dest="eval_checkpoints", action="store_false",
                        help="Do not evaluate each checkpoint on the entire train/test set. This can speed up training"
                             " but the downside is that you will be relying on training batches only for tracking the"
                             " progress of training.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, allow_id=allow_id)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--st", "--smoke-test", dest="smoke_test", action="store_true",
                        help="Conduct a quick, full test of the training pipeline. If enabled, then a number of"
                             " arguments will be overridden to make the training run as short as possible and print in"
                             " verbose/debug mode.")
    return parser


def prep_config(parser, args):
    """ Process command line arguments to produce a full training config. May also edit the arguments. """
    # If we're doing a smoke test, then we need to modify the verbosity before configuring the logger.
    if args.smoke_test and args.verbose < 2:
        args.verbose = 2

    argutils.configure_logging(args, level=logging.INFO)

    # This list governs which _top-level_ args can be overridden from the command line.
    config = argutils.load_config_from_args(parser, args, ["arch", "data_path", "save_checkpoints", "eval_checkpoints",
                                                           "device", "id", "project", "entity", "group", "verbose"])
    if not config.get("train_config"):
        # Exits the program with a usage error.
        parser.error(f'The given config does not have a "train_config" sub-config: {args.config}')
    # This list governs which _training_ args can be overridden from the command line.
    config["train_config"] = argutils.override_from_command_line(config["train_config"], parser, args,
                                                                 ["benchmark", "dataset", "seed"])

    # Conduct a quick test.
    if args.smoke_test:
        config["save_checkpoints"] = True
        config["eval_checkpoints"] = True
        config["checkpoint_initial_model"] = False
        config["train_config"]["max_steps"] = 1
        config["train_config"]["epochs"] = 1

    return config


def setup_and_train(parser, config):
    """ Setup W&B, load data, and commence training. """
    device = argutils.get_device(parser, config)
    argutils.set_seed(config["train_config"]["seed"])
    argutils.prepare_wandb(config)
    train_data, test_data, num_classes = datasets.load_dataset_from_config(config)

    logging.info("Commencing training.")
    arch = benchmark.from_string(config["arch"])
    benchmark.train_and_store(arch)
    logging.info("Training complete.")


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def main(argv=None):
    parser = create_arg_parser()
    args = parser.parse_args(argv)
    argutils.configure_logging(args)
    print_memory_stats()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed the process. This will turn on the CUDNN deterministic setting, which "
                      "can slow it down considerably!")

    if args.gpu is not None:
        warnings.warn(f"You have chosen a specific GPU ({args.gpu}). This will completely disable data parallelism.")
        if not torch.cuda.is_available():
            raise RuntimeError(f"You requested GPU {args.gpu}, but no GPUs were found.")
    elif not torch.cuda.is_available():
        if args.cpu:
            warnings.warn("Using CPU; this will be slow.")
        else:
            raise RuntimeError(f"No GPUs found. To run using CPU only, you must explicitly allow it with --cpu.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.ddp

    ngpus_per_node = torch.cuda.device_count()
    print(f"Found {ngpus_per_node} GPUs.")
    if args.ddp:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            # Setting the GPU explicitly will disable parallelism.
            args.gpu = 0
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    argutils.configure_logging(args, proc=gpu if args.ddp else None)
    args.gpu = gpu
    print_memory_stats(gpu)

    # Configure distributed processing, if using.
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.ddp:
            # For multiprocessing distributed operation, rank needs to be the global rank among all the processes.
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

        if args.gpu is not None:
            # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    rank_text = "." if args.rank == -1 else f" in process {args.rank}."
    if args.gpu is not None:
        print(f"Using GPU {args.gpu}" + rank_text)

    # Create the datasets and loaders.
    train_data, test_data, num_classes = make_datasets(args.dataset, args.data_path, args.batch_size * ngpus_per_node,
                                                       args.max_batches)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        test_sampler = None

    print(f"Using {args.workers} workers for data loading" + rank_text)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=args.workers, pin_memory=False,
                              persistent_workers=args.workers > 1)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=test_sampler is None, sampler=test_sampler,
                             num_workers=args.workers, pin_memory=False, persistent_workers=args.workers > 1)

    # Create model.
    print(f"Constructing {'pre-trained ' if args.pretrained else ''}'{args.arch}'" + rank_text)
    orig_model = models.__dict__[args.arch](pretrained=args.pretrained)

    # Wrap in ProbedModule.
    device = "cpu" if args.gpu is None else args.gpu
    orig_model.to(device)
    pm = probed_model(orig_model, args.arch, test_loader, num_classes, device)
    print_memory_stats(args.rank)

    # Push model to device.
    # TODO: Since we're not doing any backward(), I'm not sure we need to wrap the model in DataParallel structures.
    model = pm
    if not torch.cuda.is_available():
        print("Using CPU; this will be slow.")
    elif args.distributed:
        print("Using DistributedDataParallel" + rank_text)
        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device
        # scope, otherwise, DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all available GPUs if device_ids are not set.
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("Using DataParallel" + rank_text)
        # DataParallel will divide and allocate batch_size to all available GPUs.
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print_memory_stats(args.rank)

    # Finally, process all the data!
    loaders = {"Train": train_loader, "Test": test_loader}
    cached_activations, cached_labels = training.cache_probe_inputs(model, pm.probes, loaders, device,
                                                                    print_freq=args.print_freq)

    # Finally, we cache the activations for each probe.
    all_labels = {}
    for split in ("Train", "Test"):
        all_labels[split] = torch.concat(cached_labels[split])
        if args.ddp:
            all_labels[split] = gather_tensor(all_labels[split], args.world_size, args.rank)
    for pname in pm.probes:
        # Collect and concatenate all this probe's tensors.
        psets = {}
        for split in ("Train", "Test"):
            inputs = torch.concat(cached_activations[pname][split])
            # If we are distributed, we need to gather all tensors into the main process.
            if args.ddp:
                inputs = gather_tensor(inputs, args.world_size, args.rank)
            psets[split] = (inputs, all_labels[split])

        # Save to disk.
        if not args.ddp or args.rank == 0:
            if args.output:
                dest_file = args.output / activation_cache_file_name(args.arch, args.pretrained, pname)
            else:
                dest_file = activation_cache_file(args.data_path, args.dataset, args.arch, args.pretrained, pname)
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"Saving activations: {dest_file}")
            for split, (inputs, labels) in psets.items():
                print(f"    {split}: inputs = {inputs.shape} labels = {labels.shape}")
            with open(dest_file, "wb") as f:
                pickle.dump(psets, f)


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args = parser.parse_args(argv)

    config = prep_config(parser, args)

    setup_and_train(parser, config)


if __name__ == "__main__":
    sys.exit(main())
