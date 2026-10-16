"""
YAML config support for the argparse-based icefall scripts.

  python zipformer/train.py --config conf/xxx.yaml [--some-option value ...]

Values in the YAML file become the parser defaults; options given on the
command line override them. Keys may use '-' or '_' ("max-duration" ==
"max_duration"); unknown keys are an error. Lists are joined with commas, so
`chunk_size: [16, 32, 64, -1]` is the same as "16,32,64,-1".
"""
import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import yaml


def _normalize_values(parser: argparse.ArgumentParser, cfg: dict) -> None:
    """In-place: join lists with commas; stringify numbers for options whose
    default is a string (e.g. --query-head-dim, parsed later with split(","))."""
    dest2action = {a.dest: a for a in parser._actions}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)):
            cfg[k] = ",".join(str(x) for x in v)
        elif (
            isinstance(dest2action[k].default, str)
            and not isinstance(v, (str, bool))
            and v is not None
        ):
            cfg[k] = str(v)


def parse_args_with_config(
    parser: argparse.ArgumentParser, argv: Optional[List[str]] = None
) -> argparse.Namespace:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file with option values; command-line options override it.",
    )
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)

    if known.config is not None:
        with open(known.config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg = {str(k).replace("-", "_"): v for k, v in cfg.items()}
        valid = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - valid)
        if unknown:
            parser.error(f"Unknown keys in {known.config}: {unknown}")
        _normalize_values(parser, cfg)
        # String values still go through each option's `type` (argparse
        # converts string defaults), e.g. exp_dir -> Path.
        parser.set_defaults(**cfg)

    return parser.parse_args(argv)


def save_config(args: argparse.Namespace, exp_dir: Path, resume: bool) -> Path:
    """Write the fully resolved options to exp_dir/config.yaml (or a
    timestamped file when resuming, so the original is kept)."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"config-resume-{datetime.now():%Y%m%d-%H%M%S}.yaml"
        if resume
        else "config.yaml"
    )
    out = exp_dir / name
    resolved = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(args).items())
    }
    with open(out, "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, default_flow_style=False)
    return out


# Options taken from a training config when decoding, besides the model
# architecture options. chunk_size / left_context_frames are excluded: at
# decoding time they select the latency and must be single values.
TRAIN_CONFIG_EXTRA_KEYS = {"bpe_model", "context_size", "exp_dir"}
TRAIN_CONFIG_EXCLUDED_KEYS = {"chunk_size", "left_context_frames"}


def parse_args_with_train_config(
    parser: argparse.ArgumentParser,
    add_model_arguments: Callable[[argparse.ArgumentParser], None],
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:
    """For decoding scripts: `--train-config exp/cfgN/config.yaml` (the
    resolved config saved by train.py) supplies the model architecture,
    bpe_model, context_size and exp_dir, so the decoder builds exactly the
    trained model. All other training options in the file are ignored.
    Command-line options still override."""
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Resolved training config (<exp_dir>/config.yaml) to take the "
        "model architecture, bpe_model, context_size and exp_dir from.",
    )
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train-config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)

    if known.train_config is not None:
        with open(known.train_config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg = {str(k).replace("-", "_"): v for k, v in cfg.items()}
        tmp = argparse.ArgumentParser(add_help=False)
        add_model_arguments(tmp)
        wanted = ({a.dest for a in tmp._actions} | TRAIN_CONFIG_EXTRA_KEYS)
        wanted -= TRAIN_CONFIG_EXCLUDED_KEYS
        valid = {a.dest for a in parser._actions}
        cfg = {k: v for k, v in cfg.items() if k in wanted and k in valid}
        _normalize_values(parser, cfg)
        parser.set_defaults(**cfg)

    return parser.parse_args(argv)
