#!/usr/bin/env python3
"""
TiRA ablation: tail head without alignment gate
------------------------------------------------

This script provides an ablation of the TARP (Tail Aligned Routing for PEFT) algorithm
where the dual‐head hypernetwork is retained (there is still a separate tail head),
but the cosine alignment gate used to decide whether tail updates should be merged
into the shared head or kept separate is disabled.  In this ablation, all updates
coming from clients designated as part of the tail set are *always* merged into
the shared head.  Consequently, the tail head is effectively unused: clients may
train into the tail head locally, but on the server side those updates are
discarded and only their shared head contributions are aggregated.

The implementation is a thin wrapper around the original TARP training loop.  It
constructs the same hypernetwork and client objects but overrides the routing
threshold to a very low value so that every tail update is considered aligned and
merged into the shared head.  No other changes are made to the code – the risk
tracking and tail set selection logic remain identical to the original.  This
script is intended for experiments in which the impact of the alignment gate is
isolated.

Usage example:

    python tira_no_alignment_gate.py --num_clients 20 --clients_per_round 6 --ft_frac 0.30

All command line arguments available in the original TARP script remain valid.
See the documentation in ``tarp.py`` for more details.
"""

import argparse
import time
import os
import random
import json
import math
from typing import List

import numpy as np

import torch

import flwr as fl
from transformers import AutoProcessor, Wav2Vec2ForCTC

from torch.utils.data import DataLoader
from tarp import filter_langs_supported_by_mms
import splits as splits_mod
# The code below is largely identical to the reference implementation in
# ``tarp.py``.  It is reproduced here in a simplified form
# to avoid a hard dependency on that file, and to allow us to override the
# alignment gate threshold cleanly.  See the original script for detailed
# comments on each component.


################################################################################
#                               Utility functions                              #
################################################################################

def tail_mean(values: List[float], tail_frac: float, higher_is_worse: bool) -> float:
    """Return the mean of either the worst ``tail_frac`` values (if higher_is_worse) or
    the best ``tail_frac`` values.  Used for reporting tail performance metrics."""
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return float("nan")
    xs.sort()
    k = max(1, int(math.ceil(tail_frac * len(xs))))
    tail = xs[-k:] if higher_is_worse else xs[:k]
    return float(sum(tail) / len(tail))

def pctl(values: List[float], q: float) -> float:
    """Compute the q‑th percentile of ``values`` (0 ≤ q ≤ 100)."""
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return float("nan")
    xs.sort()
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])

################################################################################
#                               Hypernetwork code                              #
################################################################################

# A minimal HyperLoRA implementation is included here for completeness.  The
# original TARP script defines many helper functions and classes to construct
# a dual‑head hypernetwork.  We reuse that code unchanged except that we
# explicitly set the ``route_cos_thr`` parameter when constructing the server
# strategy.  The hypernetwork uses the shared head for most clients and the
# tail head for a subset of clients as determined by the server.


from tarp import (
    DualHeadHyperLoRA,
    HyperLoRALinear,
    HyperContext,
    LocalFleursIterable,
    CTCCollator,
    LocalFleursTARPClient,
    _find_hidden_size_and_n_layers,
    get_trainable_weights,
    ndarrays_to_parameters,
    set_trainable_weights,
    parameters_to_ndarrays,
    bytes_of_parameters,
    l2_delta,
)

from tarp import TARPFedAvg as _BaseTARPFedAvg


class NoAlignTailFedAvg(_BaseTARPFedAvg):
    """FedAvg strategy that disables the alignment gate.

    This subclass of the reference ``TARPFedAvg`` simply forces the
    ``route_cos_thr`` to a very low value so that every tail update is merged
    into the shared head.  All other behaviour (risk tracking, tail set
    selection, logging) is inherited from the base class.
    """

    def __init__(self, *args, **kwargs):
        # We explicitly override the cosine threshold here.  Any value below
        # -1.0 suffices because cosine similarity lies in [-1, 1]; setting it to
        # -2.0 ensures that all tail updates are considered aligned.
        kwargs = dict(kwargs)
        kwargs["route_cos_thr"] = -2.0
        super().__init__(*args, **kwargs)


def run_experiment(args):
    """Run a federated training experiment with the no‑alignment ablation."""

    # Initialise random seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load and filter languages based on MMS support.  We delegate to the
    # reference script's helper functions to build manifests and resolve the
    # mapping from language ID to MMS tokenizer targets.


    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root not found: {args.data_root}")

    exclude = {x.strip() for x in args.exclude_langs.split(",") if x.strip()}

    # Determine which device to use for local training.  If fp16 is requested
    # explicitly use CUDA when available, otherwise fall back to CPU.
    if args.no_fp16:
        fp16 = False
    elif args.fp16:
        fp16 = True
    else:
        fp16 = args.device.startswith("cuda")

    if args.no_mms_adapters:
        use_mms_adapters = False
    elif args.use_mms_adapters:
        use_mms_adapters = True
    else:
        use_mms_adapters = True

    # Select the set of languages/clients.  If an explicit list of languages is
    # provided it will be honoured; otherwise we discover languages in
    # ``data_root``.  The number of clients may be overridden via
    # ``--num_clients``.
    if args.num_clients is not None:
        n_clients = int(args.num_clients)
    else:
        n_clients = int(args.num_langs)

    if args.langs.strip():
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
        if n_clients > 0:
            langs = langs[:n_clients]
    else:
        discovered = splits_mod.discover_langs(args.data_root, exclude=exclude)
        langs = discovered[: max(1, n_clients)] if n_clients > 0 else discovered

    langs = [l for l in langs if l not in exclude]
    if not langs:
        raise RuntimeError("No languages selected")

    ok_langs = splits_mod.build_manifests_for_langs(
        data_root=args.data_root,
        langs=langs,
        force_rebuild=args.force_rebuild_manifests,
        fail_fast=args.fail_fast,
    )

    ok_langs, lang_to_mms = filter_langs_supported_by_mms(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        data_langs=ok_langs,
    )
    if not ok_langs:
        raise RuntimeError("No usable languages after MMS mapping filter")

    # Optionally write the list of languages to disk for reproducibility.
    if args.write_langs_json:
        with open(args.write_langs_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_root": args.data_root,
                    "langs": ok_langs,
                    "num_clients": len(ok_langs),
                    "ft_frac": float(args.ft_frac),
                    "route_tail_frac": float(args.route_tail_frac),
                },
                f,
                indent=2,
            )

    if args.build_manifests_only:
        return

    all_langs = ok_langs

    clients_per_round = int(args.clients_per_round)
    if clients_per_round > len(all_langs):
        print(
            f"[warn] clients_per_round={clients_per_round} > num_clients={len(all_langs)}; clamping."
        )
        clients_per_round = len(all_langs)

    eval_utterances = int(args.eval_utterances)
    local_steps = int(args.local_steps)
    max_train_hours = float(args.max_train_hours)
    max_valid_hours = float(args.max_valid_hours)
    ft_frac_use = float(args.ft_frac)

    if args.unbounded_budget_baseline:
        clients_per_round = len(all_langs)
        ft_frac_use = 1.0
        max_train_hours = 0.0
        max_valid_hours = 0.0
        eval_utterances = 0
        local_steps = 0

    # Initialise global hypernetwork parameters.  We create a temporary model to
    # determine the hidden size and number of encoder layers, and then build a
    # DualHeadHyperLoRA.  Only the hypernetwork parameters are trainable; the
    # base Wav2Vec2 model remains frozen.
    tmp_model = Wav2Vec2ForCTC.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    hidden, n_layer = _find_hidden_size_and_n_layers(tmp_model)
    tmp_hyper = DualHeadHyperLoRA(n_lang=len(all_langs), n_layer=n_layer, hidden_size=hidden).to("cpu")
    for p in tmp_hyper.parameters():
        p.requires_grad = True
    trainable_names, init_arrays = get_trainable_weights(tmp_hyper)
    init_parameters = ndarrays_to_parameters(init_arrays)

    def client_fn(cid: str):
        i = int(cid)
        lang_id = all_langs[i]
        mms_target = lang_to_mms.get(lang_id, None)

        return LocalFleursTARPClient(
            cid=cid,
            lang_id=lang_id,
            lang_dir=os.path.join(args.data_root, lang_id),
            lang_idx=i,
            all_langs=all_langs,
            mms_target_lang=mms_target,
            model_id=args.model_id,
            device=args.device,
            fp16=fp16,
            use_mms_adapters=use_mms_adapters,
            batch_size=args.batch_size,
            local_steps=local_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            ft_frac=ft_frac_use,
            max_train_hours=max_train_hours,
            max_valid_hours=max_valid_hours,
            max_audio_s=args.max_audio_s,
            eval_utterances=eval_utterances,
            trainable_names=trainable_names,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    # Use our NoAlignTailFedAvg strategy.  We intentionally pass a
    # ``route_cos_thr`` value so low that the alignment gate never filters any
    # tail updates.  All other routing parameters are forwarded from the
    # command line.
    strategy = NoAlignTailFedAvg(
        trainable_names=trainable_names,
        log_csv=args.log_csv,
        tail_fracs=[float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()] or [0.10, 0.20, 0.30],
        route_tail_frac=float(args.route_tail_frac),
        # route_cos_thr is overridden in the strategy constructor to -2.0
        route_cos_thr=float(args.route_cos_thr),
        route_warmup_rounds=int(args.route_warmup_rounds),
        route_risk_beta=float(args.route_risk_beta),
        route_risk_metric=str(args.route_risk_metric),
        fraction_fit=min(1.0, clients_per_round / max(1, len(all_langs))),
        fraction_evaluate=1.0,
        min_fit_clients=min(clients_per_round, len(all_langs)),
        min_available_clients=len(all_langs),
        min_evaluate_clients=len(all_langs),
        initial_parameters=init_parameters,
    )

    # Resource allocation per client.  Adjust these values to reflect your
    # hardware (e.g. number of CPU cores and GPUs per client).
    client_resources = {
        "num_cpus": float(args.num_cpus_per_client),
        "num_gpus": float(args.num_gpus_per_client),
    }

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(all_langs),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"include_dashboard": False},
    )


def main():
    parser = argparse.ArgumentParser()
    # The argument list mirrors the original TARP script.  Only those options
    # relevant to the no‑alignment ablation are documented here; for further
    # details see ``tarp.py``.
    parser.add_argument("--data_root", type=str, default="../ML_SUPERB/fleurs")
    parser.add_argument("--langs", type=str, default="")
    parser.add_argument("--num_langs", type=int, default=20)
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--exclude_langs", type=str, default="cmn,jpn,kor,zh_cn,ja_jp,ko_kr")
    parser.add_argument("--build_manifests_only", action="store_true")
    parser.add_argument("--force_rebuild_manifests", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--write_langs_json", type=str, default="langs_used.json")
    parser.add_argument("--model_id", type=str, default="facebook/mms-1b-fl102")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--use_mms_adapters", action="store_true")
    parser.add_argument("--no_mms_adapters", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients_per_round", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ft_frac", type=float, default=0.30)
    parser.add_argument("--max_train_hours", type=float, default=0.50)
    parser.add_argument("--max_valid_hours", type=float, default=0.10)
    parser.add_argument("--max_audio_s", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--local_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_utterances", type=int, default=128)
    parser.add_argument("--log_csv", type=str, default="runs/tira_no_align.csv")
    parser.add_argument("--num_cpus_per_client", type=float, default=2.0)
    parser.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)
    parser.add_argument("--unbounded_budget_baseline", action="store_true")
    parser.add_argument("--tail_fracs", type=str, default="0.10,0.20,0.30")
    parser.add_argument("--route_tail_frac", type=float, default=0.20)
    # ``route_cos_thr`` will be ignored by our strategy but we expose it for
    # completeness.
    parser.add_argument("--route_cos_thr", type=float, default=0.0)
    parser.add_argument("--route_warmup_rounds", type=int, default=10)
    parser.add_argument("--route_risk_beta", type=float, default=0.90)
    parser.add_argument("--route_risk_metric", type=str, default="loss", choices=["cer", "loss"])

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()