#!/usr/bin/env python3
"""
TiRA ablation: alignment gate without tail separation
----------------------------------------------------

This script implements a second ablation of the Tail Aligned Routing Protocol
(TARP) in which the dual‐head hypernetwork is collapsed into a single branch.
In the original TARP algorithm, clients deemed at risk (the "tail" set) train
on a separate tail head, and at aggregation time the server decides—based on
cosine similarity—whether to merge their updates into the shared head or to
keep them isolated in the tail head.  In this ablation the tail head is
removed entirely: all clients use the shared head for local training, and the
server uses the alignment gate to determine whether each client's update
should be incorporated into the global model.  Updates from clients whose
direction is misaligned with the reference update are filtered out entirely.

The implementation here reuses most of the original TARP components.  The
critical change is the server strategy: ``AlignNoTailFedAvg`` overrides the
routing logic to disable the tail head, enforce that all clients train on the
shared head, and drop misaligned updates.  No per‑client configuration is
needed beyond the standard hypernetwork injection.

Usage example:

    python tira_no_tail_separation.py --num_clients 20 --clients_per_round 6 --ft_frac 0.30

Command line arguments mirror those of the original TARP script.  Only the
routing parameters (``--route_tail_frac`` and ``--route_cos_thr``) influence
the behaviour of the alignment gate in this ablation.  See
``tarp_hyperlora_fl_fleurs.py`` for more details on the underlying model and
training loop.
"""

import argparse
import os
import json
import time
import random
import math
from typing import List, Dict, Tuple

import numpy as np

import torch

import flwr as fl

from transformers import Wav2Vec2ForCTC

# Import components from the reference implementation.  This includes the
# hypernetwork classes, client implementation and helper functions.  The
# ``TARPFedAvg`` strategy is used as a base class for our custom server.
from tarp_hyperlora_fl_fleurs import (
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
    tail_mean,
    pctl,
    cosine_sim,
)
from tarp_hyperlora_fl_fleurs import TARPFedAvg as _BaseTARPFedAvg

# Additional helpers from the reference script
from tarp_hyperlora_fl_fleurs import filter_langs_supported_by_mms
from tarp_hyperlora_fl_fleurs import splits as splits_mod


class AlignNoTailFedAvg(_BaseTARPFedAvg):
    """FedAvg strategy with alignment gate but without tail separation.

    In this variant of TARP the server no longer maintains a separate tail
    head.  All clients train on the shared head.  At aggregation time the
    server computes a reference update direction from the non‑tail clients and
    compares each tail client's update against this reference using cosine
    similarity.  Clients whose updates are aligned (above the threshold) are
    included in the aggregation; those whose updates are misaligned are
    dropped entirely.  No updates are applied to the tail head because it
    does not exist in this ablation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Disable the use of the tail head entirely.  All clients will always use
    # the shared head; the tail set is still computed for alignment gating
    # purposes but ``use_tail_head`` will be False for every client.
    def _cfg_use_tail_head(self, server_round: int, lang_idx: int) -> bool:
        return False

    def configure_fit(self, server_round, parameters, client_manager):
        # Start per-round timer
        self._round_start_ts = time.time()

        cfgs = super().configure_fit(server_round, parameters, client_manager)

        # Cache payload size and sent ndarrays for delta calculation
        self._clients_fit_configured_this_round = int(len(cfgs))
        if len(cfgs) > 0:
            fit_ins0 = cfgs[0][1]
            self._payload_bytes_this_round = float(bytes_of_parameters(fit_ins0.parameters))
            try:
                self._sent_ndarrays_this_round = parameters_to_ndarrays(fit_ins0.parameters)
            except Exception:
                self._sent_ndarrays_this_round = None
        else:
            self._payload_bytes_this_round = float("nan")
            self._sent_ndarrays_this_round = None

        # Ensure ``use_tail_head`` is disabled in the client configuration
        out = []
        for client, fit_ins in cfgs:
            new_cfg = dict(fit_ins.config)
            new_cfg["use_tail_head"] = 0
            out.append((client, FitIns(fit_ins.parameters, new_cfg)))
        return out

    def configure_evaluate(self, server_round, parameters, client_manager):
        cfgs = super().configure_evaluate(server_round, parameters, client_manager)
        out = []
        for client, ev_ins in cfgs:
            new_cfg = dict(ev_ins.config)
            new_cfg["use_tail_head"] = 0
            out.append((client, EvaluateIns(ev_ins.parameters, new_cfg)))
        return out

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print(f"[warn] Round {server_round}: no fit results, keeping previous parameters")
            return (self._last_parameters, {}) if self._last_parameters is not None else None

        # Fit-side statistics (as in the base class)
        clients_fit = int(len(results))
        fit_losses: List[float] = []
        delta_l2s: List[float] = []

        sent_nds = self._sent_ndarrays_this_round
        payload_bytes = float(self._payload_bytes_this_round) if self._payload_bytes_this_round == self._payload_bytes_this_round else float("nan")

        for _, fit_res in results:
            m = fit_res.metrics or {}
            if "train_loss" in m:
                try:
                    fit_losses.append(float(m["train_loss"]))
                except Exception:
                    pass
            if sent_nds is not None:
                try:
                    client_nds = parameters_to_ndarrays(fit_res.parameters)
                    d = l2_delta(client_nds, sent_nds)
                    if not math.isnan(d):
                        delta_l2s.append(float(d))
                except Exception:
                    pass

        fit_mean_train_loss = float(np.mean(fit_losses)) if len(fit_losses) > 0 else float("nan")
        fit_p95_train_loss = float(np.percentile(fit_losses, 95)) if len(fit_losses) > 0 else float("nan")
        fit_mean_delta_l2 = float(np.mean(delta_l2s)) if len(delta_l2s) > 0 else float("nan")
        fit_max_delta_l2 = float(np.max(delta_l2s)) if len(delta_l2s) > 0 else float("nan")

        if payload_bytes == payload_bytes:
            round_download_bytes = float(payload_bytes * float(self._clients_fit_configured_this_round))
            round_upload_bytes = float(payload_bytes * float(clients_fit))
        else:
            round_download_bytes = float("nan")
            round_upload_bytes = float("nan")

        self._fit_round_stats[int(server_round)] = {
            "clients_fit": float(clients_fit),
            "fit_mean_train_loss": float(fit_mean_train_loss),
            "fit_p95_train_loss": float(fit_p95_train_loss),
            "fit_mean_delta_l2": float(fit_mean_delta_l2),
            "fit_max_delta_l2": float(fit_max_delta_l2),
            "payload_bytes": float(payload_bytes),
            "round_upload_bytes": float(round_upload_bytes),
            "round_download_bytes": float(round_download_bytes),
        }

        # Obtain current global parameters
        if self._last_parameters is not None:
            global_nds = parameters_to_ndarrays(self._last_parameters)
        else:
            global_nds = parameters_to_ndarrays(self.initial_parameters) if hasattr(self, "initial_parameters") else None
        if global_nds is None:
            global_nds = parameters_to_ndarrays(results[0][1].parameters)

        self._last_nds = [a.copy() for a in global_nds]

        # Partition clients into shared and tail groups according to the tail set
        shared_group: List[Tuple[float, List[np.ndarray], int]] = []
        tail_group: List[Tuple[float, List[np.ndarray], int]] = []
        for _, fit_res in results:
            w = float(getattr(fit_res, "num_examples", 1.0))
            nds = parameters_to_ndarrays(fit_res.parameters)
            m = fit_res.metrics or {}
            lang_idx = int(m.get("lang_idx", -1))
            # Clients in ``self._tail_lang_idxs`` are considered tail
            if lang_idx in self._tail_lang_idxs:
                tail_group.append((w, nds, lang_idx))
            else:
                shared_group.append((w, nds, lang_idx))

        # Reference update direction from shared clients
        ref_vec = None
        if self._idx_shared_head:
            deltas = []
            for w, nds, _ in shared_group:
                # Flatten the shared head delta for each client
                dv = _flatten_params([nds[i] - global_nds[i] for i in range(len(nds))], self._idx_shared_head)
                deltas.append((w, dv))
            if deltas:
                ref_vec = _avg_ndarrays(deltas)

        # Determine which tail updates to include based on cosine similarity
        merged_tail: List[Tuple[float, List[np.ndarray], int]] = []
        kept_tail: List[Tuple[float, List[np.ndarray], int]] = []
        tail_cos: List[float] = []
        if ref_vec is None:
            # No reference (e.g. no shared clients); reject all tail updates
            kept_tail = tail_group[:]
            tail_cos = [0.0 for _ in tail_group]
        else:
            for w, nds, lang_idx in tail_group:
                # Compute the update direction for the tail head
                dv_tail = _flatten_params([nds[i] - global_nds[i] for i in range(len(nds))], self._idx_tail_head)
                c = cosine_sim(dv_tail, ref_vec)
                tail_cos.append(c)
                if c >= self.route_cos_thr:
                    merged_tail.append((w, nds, lang_idx))
                else:
                    kept_tail.append((w, nds, lang_idx))

        # Store cosine statistics for logging
        self._round_cos_stats = {
            "merged_tail_into_shared": float(len(merged_tail)),
            "kept_tail_in_tailhead": float(len(kept_tail)),
            "mean_tail_cos": float(sum(tail_cos) / max(1, len(tail_cos))) if tail_cos else float("nan"),
            "min_tail_cos": float(min(tail_cos)) if tail_cos else float("nan"),
        }

        # Clients whose updates will be applied
        accepted = shared_group + merged_tail

        n_params = len(global_nds)
        new_nds = [a.copy() for a in global_nds]

        # Aggregate each parameter across the accepted clients.  If no clients are
        # accepted, we simply keep the global parameter for that index.
        for j in range(n_params):
            if accepted:
                weighted = [(w, nds[j]) for (w, nds, _) in accepted]
                new_nds[j] = _avg_ndarrays(weighted)
            else:
                new_nds[j] = global_nds[j].copy()

        # Update the language embedding row for each accepted client
        if self._idx_lang_emb:
            li = self._idx_lang_emb[0]
            lang_emb = new_nds[li].copy()
            for _, nds, lang_idx in accepted:
                if 0 <= lang_idx < lang_emb.shape[0]:
                    lang_emb[lang_idx] = nds[li][lang_idx]
            new_nds[li] = lang_emb

        params_fixed = ndarrays_to_parameters(new_nds)
        self._last_parameters = params_fixed
        return params_fixed, {}

    # ``aggregate_evaluate`` remains unchanged from the base class; we reuse it
    # verbatim so that risk tracking and tail set selection work as before.


def run_experiment(args):
    """Run the federated learning experiment with no tail separation."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root not found: {args.data_root}")

    exclude = {x.strip() for x in args.exclude_langs.split(",") if x.strip()}

    # Determine FP16 usage
    if args.no_fp16:
        fp16 = False
    elif args.fp16:
        fp16 = True
    else:
        fp16 = args.device.startswith("cuda")

    # Determine use of MMS adapters
    if args.no_mms_adapters:
        use_mms_adapters = False
    elif args.use_mms_adapters:
        use_mms_adapters = True
    else:
        use_mms_adapters = True

    # Determine number of clients
    if args.num_clients is not None:
        n_clients = int(args.num_clients)
    else:
        n_clients = int(args.num_langs)

    # Language selection
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

    if args.write_langs_json:
        with open(args.write_langs_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_root": args.data_root,
                    "langs": ok_langs,
                    "num_clients": len(ok_langs),
                    "ft_frac": float(args.ft_frac),
                    "route_tail_frac": float(args.route_tail_frac),
                    "route_cos_thr": float(args.route_cos_thr),
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

    # Initialize global hypernetwork parameters
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

    tail_fracs = [float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()]
    if not tail_fracs:
        tail_fracs = [0.10, 0.20, 0.30]

    strategy = AlignNoTailFedAvg(
        trainable_names=trainable_names,
        log_csv=args.log_csv,
        tail_fracs=tail_fracs,
        route_tail_frac=float(args.route_tail_frac),
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
    parser.add_argument("--log_csv", type=str, default="runs/tira_no_tail.csv")
    parser.add_argument("--num_cpus_per_client", type=float, default=2.0)
    parser.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)
    parser.add_argument("--unbounded_budget_baseline", action="store_true")
    parser.add_argument("--tail_fracs", type=str, default="0.10,0.20,0.30")
    parser.add_argument("--route_tail_frac", type=float, default=0.20)
    parser.add_argument("--route_cos_thr", type=float, default=0.0)
    parser.add_argument("--route_warmup_rounds", type=int, default=10)
    parser.add_argument("--route_risk_beta", type=float, default=0.90)
    parser.add_argument("--route_risk_metric", type=str, default="loss", choices=["cer", "loss"])

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()