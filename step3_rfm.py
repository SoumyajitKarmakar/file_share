"""
Step 3: RFM probe with OOD evaluation.

Trains an RFM probe on v2 (country-capital) and evaluates on:
  - v2 in-domain test split
  - element-symbol dataset (OOD)

Mirrors step2_ood.py's structure exactly so numbers are directly comparable.
Same SEED, same fact splits, same per-condition AUC metrics.

RFM fitting follows train_rfm_probe_on_concept() in direction_utils.py:
  - kernel='l2_high_dim'
  - hyperparameter sweep over (bandwidth, reg, center_grads)
  - selects by validation AUC

Reuses cached activations from step2_ood.py: activations_v2.pt, activations_ood_elements.pt.




python step3_rfm.py --layers_subset -1,-11,-20,-27

python step3_rfm.py

python step3_rfm.py \
    --bws 10 \
    --regs 1e-3 \
    --center_grads true \
    --v2_dataset conflict_dataset_v2.json \
    --ood_dataset conflict_dataset_ood_elements.json \
    --v2_cache all_results/ \
    --ood_cache all_results/ \
    --lr_results all_results/ \
    --out_dir all_results/



--n_components


"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from xrfm import RFM

from step1_lr_baseline import (
    SEED,
    load_dataset,
    split_by_fact,
)

print(f"Running with seed: {SEED}")

CACHE_DIR = "/scratch/bbjr/skarmakar/huggingface"

# -------------------- RFM fitting --------------------

def fit_rfm(train_X, train_y, val_X, val_y,
            regs, bws, center_grads,
            rfm_iters, n_components,
            tuning_metric='auc',
            device='cuda'):
    """
    Hyperparameter sweep for RFM. Returns the model with the best val AUC, the
    best params dict, and the best val score.

    Pattern follows train_rfm_probe_on_concept in direction_utils.py.
    """
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)

    best_model = None
    best_params = None
    best_score = float('-inf')

    for reg in regs:
        for bw in bws:
            for cg in center_grads:
                try:
                    model = RFM(
                        kernel='l2_high_dim',
                        bandwidth=bw,
                        tuning_metric=tuning_metric,
                        device=device,
                    )
                    model.fit(
                        (train_X, train_y),
                        (val_X, val_y),
                        reg=reg,
                        iters=rfm_iters,
                        center_grads=cg,
                        early_stop_rfm=True,
                        get_agop_best_model=True,
                        top_k=n_components,
                    )

                    val_preds = model.predict(val_X)
                    if torch.is_tensor(val_preds):
                        val_preds = val_preds.cpu().numpy()
                    val_preds = np.asarray(val_preds).ravel()
                    val_score = float(roc_auc_score(val_y.cpu().numpy().ravel(), val_preds))

                    if val_score > best_score:
                        best_score = val_score
                        best_model = deepcopy(model)
                        best_params = {'reg': reg, 'bw': bw, 'center_grads': bool(cg)}
                except Exception as e:
                    print(f"    [warn] RFM fit failed for reg={reg}, bw={bw}, cg={cg}: "
                          f"{type(e).__name__}: {e}")
                    continue

    return best_model, best_params, best_score


# -------------------- evaluation helpers --------------------

def model_score(model, X, device='cuda'):
    """Return 1D numpy array of model predictions on X."""
    preds = model.predict(X.to(device))
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    return np.asarray(preds).ravel()


def per_condition_scores(H, idx_t1, idx_t2, idx_t3, idx_t6, model):
    return (
        model_score(model, H[idx_t1]),
        model_score(model, H[idx_t2]),
        model_score(model, H[idx_t3]),
        model_score(model, H[idx_t6]),
    )


def three_aucs(s1, s2, s3, s6):
    def auc(pos, neg):
        try:
            return float(roc_auc_score(
                np.concatenate([np.ones_like(pos), np.zeros_like(neg)]),
                np.concatenate([pos, neg]),
            ))
        except ValueError:
            return float('nan')
    id_auc = auc(s3, s1)                                    # t3 vs t1
    tgt_auc = auc(s3, s6)                                   # t3 vs t6
    mon_auc = auc(s3, np.concatenate([s1, s2, s6]))         # t3 vs t1+t2+t6
    return id_auc, tgt_auc, mon_auc


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v2_dataset', type=str, default='conflict_dataset_v2.json')
    parser.add_argument('--ood_dataset', type=str, default='conflict_dataset_ood_elements.json')
    parser.add_argument('--v2_cache', type=str, default='activations_v2.pt')
    parser.add_argument('--ood_cache', type=str, default='activations_ood_elements.pt')
    parser.add_argument('--lr_results', type=str, default='step2_ood_results/results.json',
                        help="Step 2 LR results JSON. If present, the summary print shows LR vs RFM side by side.")
    parser.add_argument('--out_dir', type=str, default='step3_rfm_results')
    parser.add_argument('--rfm_iters', type=int, default=8)
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--bws', type=str, default='1,10,100',
                        help="Comma-separated bandwidths for RFM hyperparam search.")
    parser.add_argument('--regs', type=str, default='1e-3',
                        help="Comma-separated reg values for RFM hyperparam search.")
    parser.add_argument('--center_grads', type=str, default='true,false',
                        help="Comma-separated booleans for center_grads (true/false).")
    parser.add_argument('--tuning_metric', type=str, default='auc')
    parser.add_argument('--layers_subset', type=str, default=None,
                        help="Optional comma-separated layers to run (e.g. '-1,-11,-22'). "
                             "If omitted, runs all layers.")
    args = parser.parse_args()

    bws = [float(x) for x in args.bws.split(',')]
    regs = [float(x) for x in args.regs.split(',')]
    center_grads = [x.strip().lower() == 'true' for x in args.center_grads.split(',')]

    out_dir = Path(args.out_dir) / f"S{SEED}" / "step3_rfm_results"
    out_dir.mkdir(exist_ok=True, parents=True)

    # ---- load activations (must already be cached from step2) ----
    v2_cache = Path(args.v2_cache) / f"S{SEED}" / "activations.pt"
    ood_cache = Path(args.ood_cache) / f"S{SEED}" / "activations_ood_elements.pt"
    if not v2_cache.exists() or not ood_cache.exists():
        raise FileNotFoundError(
            f"Activation caches missing. Run step2_ood.py first to generate "
            f"{v2_cache} and {ood_cache}."
        )

    print(f"Loading v2 activations from {v2_cache}")
    v2_acts = torch.load(v2_cache, weights_only=True)
    print(f"Loading OOD activations from {ood_cache}")
    ood_acts = torch.load(ood_cache, weights_only=True)

    # Optional LR comparison
    lr_results = None
    lr_path = Path(args.lr_results) / f"S{SEED}" / 'step2_ood_results' / f'results_step2_S{SEED}.json'
    if args.lr_results and lr_path.exists():
        with open(lr_path) as f:
            lr_results = json.load(f)
        print(f"Loaded LR baseline from {lr_path}")
    else:
        print(f"LR results not found at {lr_path}; printing RFM-only summary.")

    # ---- splits (identical to step1/step2) ----
    v2_rows = load_dataset(args.v2_dataset)
    ood_rows = load_dataset(args.ood_dataset)
    n_v2 = len(v2_rows)
    n_ood = len(ood_rows)

    train_facts, val_facts, test_facts = split_by_fact(n_v2, seed=SEED)

    def idx_for(fact_indices, type_idx):
        return np.array([f * 4 + type_idx for f in fact_indices])

    v2_train_t1 = idx_for(train_facts, 0)
    v2_train_t3 = idx_for(train_facts, 2)
    v2_val_t1 = idx_for(val_facts, 0)
    v2_val_t3 = idx_for(val_facts, 2)
    v2_test_t1 = idx_for(test_facts, 0)
    v2_test_t2 = idx_for(test_facts, 1)
    v2_test_t3 = idx_for(test_facts, 2)
    v2_test_t6 = idx_for(test_facts, 3)

    ood_all = np.arange(n_ood)
    ood_t1 = idx_for(ood_all, 0)
    ood_t2 = idx_for(ood_all, 1)
    ood_t3 = idx_for(ood_all, 2)
    ood_t6 = idx_for(ood_all, 3)

    # ---- layer subset ----
    all_layers = sorted(v2_acts.keys(), reverse=True)
    if args.layers_subset:
        wanted = [int(x) for x in args.layers_subset.split(',')]
        layers = [L for L in all_layers if L in wanted]
        print(f"Running subset of layers: {layers}")
    else:
        layers = all_layers
        print(f"Running all {len(layers)} layers")

    print(f"\nRFM hyperparam search: bws={bws}, regs={regs}, center_grads={center_grads}")
    print(f"rfm_iters={args.rfm_iters}, n_components={args.n_components}, tuning_metric={args.tuning_metric}")
    print(f"v2: train={len(train_facts)} val={len(val_facts)} test={len(test_facts)} facts")
    print(f"OOD: {n_ood} facts (all used for OOD eval)")

    # ---- per-layer ----
    results = {
        "layers": [],
        "per_layer": {},
        "config": {
            "bws": bws, "regs": regs, "center_grads": center_grads,
            "rfm_iters": args.rfm_iters, "n_components": args.n_components,
            "tuning_metric": args.tuning_metric,
            "seed": SEED,
        },
    }

    for layer in tqdm(layers, desc="layers"):
        Hv = v2_acts[layer].float()
        Ho = ood_acts[layer].float()

        train_X = torch.cat([Hv[v2_train_t1], Hv[v2_train_t3]], dim=0)
        train_y = torch.cat([torch.zeros(len(v2_train_t1), 1), torch.ones(len(v2_train_t3), 1)], dim=0)
        val_X = torch.cat([Hv[v2_val_t1], Hv[v2_val_t3]], dim=0)
        val_y = torch.cat([torch.zeros(len(v2_val_t1), 1), torch.ones(len(v2_val_t3), 1)], dim=0)

        model, params, val_score = fit_rfm(
            train_X, train_y, val_X, val_y,
            regs=regs, bws=bws, center_grads=center_grads,
            rfm_iters=args.rfm_iters, n_components=args.n_components,
            tuning_metric=args.tuning_metric,
        )

        if model is None:
            print(f"  layer {layer}: all RFM hyperparam fits failed; skipping")
            results["per_layer"][str(layer)] = {"error": "all RFM fits failed"}
            continue

        # In-domain (v2 test)
        s_v1, s_v2, s_v3, s_v6 = per_condition_scores(
            Hv, v2_test_t1, v2_test_t2, v2_test_t3, v2_test_t6, model
        )
        id_in, tgt_in, mon_in = three_aucs(s_v1, s_v2, s_v3, s_v6)

        # OOD (all elements)
        s_o1, s_o2, s_o3, s_o6 = per_condition_scores(
            Ho, ood_t1, ood_t2, ood_t3, ood_t6, model
        )
        id_ood, tgt_ood, mon_ood = three_aucs(s_o1, s_o2, s_o3, s_o6)

        results["layers"].append(layer)
        results["per_layer"][str(layer)] = {
            "best_params": params,
            "val_auc": float(val_score),
            "in_domain": {
                "id_auc": id_in, "tgt_auc": tgt_in, "mon_auc": mon_in,
                "mean_t1": float(np.mean(s_v1)), "mean_t2": float(np.mean(s_v2)),
                "mean_t3": float(np.mean(s_v3)), "mean_t6": float(np.mean(s_v6)),
            },
            "ood": {
                "id_auc": id_ood, "tgt_auc": tgt_ood, "mon_auc": mon_ood,
                "mean_t1": float(np.mean(s_o1)), "mean_t2": float(np.mean(s_o2)),
                "mean_t3": float(np.mean(s_o3)), "mean_t6": float(np.mean(s_o6)),
            },
        }

        # Free GPU memory between layers
        del model
        torch.cuda.empty_cache()

    # ---- save ----
    out_path = out_dir / f"results_{regs}_{bws}_{center_grads}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

    # ---- summary ----
    valid_layers = [L for L in layers if str(L) in results["per_layer"]
                    and "error" not in results["per_layer"][str(L)]]
    if not valid_layers:
        print("No successful RFM fits. Check the hyperparam ranges and xrfm install.")
        return

    have_lr = lr_results is not None

    print()
    print("=" * 110)
    print("PER-LAYER: RFM vs LR")
    print("  TGT = type 3 (pos) vs type 6 (neg)   |   MON = type 3 (pos) vs type 1+2+6 (neg)")
    print("=" * 110)
    if have_lr:
        print(f"{'layer':>6}  "
              f"{'LR TGT in':>10} {'LR TGT ood':>10}   "
              f"{'RF TGT in':>10} {'RF TGT ood':>10}   "
              f"{'LR MON in':>10} {'LR MON ood':>10}   "
              f"{'RF MON in':>10} {'RF MON ood':>10}")
    else:
        print(f"{'layer':>6}  "
              f"{'RF TGT in':>10} {'RF TGT ood':>10}   "
              f"{'RF MON in':>10} {'RF MON ood':>10}")

    for layer in layers:
        r = results["per_layer"].get(str(layer))
        if not r or "error" in r:
            continue
        rfm_in, rfm_ood = r["in_domain"], r["ood"]
        if have_lr:
            lr_layer = lr_results["per_layer"].get(str(layer))
            if lr_layer:
                lr_in = lr_layer["linear"]["in_domain"]
                lr_ood = lr_layer["linear"]["ood"]
                print(f"{layer:>6d}  "
                      f"{lr_in['tgt_auc']:>10.3f} {lr_ood['tgt_auc']:>10.3f}   "
                      f"{rfm_in['tgt_auc']:>10.3f} {rfm_ood['tgt_auc']:>10.3f}   "
                      f"{lr_in['mon_auc']:>10.3f} {lr_ood['mon_auc']:>10.3f}   "
                      f"{rfm_in['mon_auc']:>10.3f} {rfm_ood['mon_auc']:>10.3f}")
            else:
                print(f"{layer:>6d}  (no LR data)")
        else:
            print(f"{layer:>6d}  "
                  f"{rfm_in['tgt_auc']:>10.3f} {rfm_ood['tgt_auc']:>10.3f}   "
                  f"{rfm_in['mon_auc']:>10.3f} {rfm_ood['mon_auc']:>10.3f}")

    # Best layers
    best_in = max(valid_layers, key=lambda L: results["per_layer"][str(L)]["in_domain"]["tgt_auc"])
    best_ood = max(valid_layers, key=lambda L: results["per_layer"][str(L)]["ood"]["tgt_auc"])
    print(f"\nRFM best layer by in-domain TGT AUC: {best_in} "
          f"(in={results['per_layer'][str(best_in)]['in_domain']['tgt_auc']:.3f}, "
          f"ood={results['per_layer'][str(best_in)]['ood']['tgt_auc']:.3f})")
    print(f"RFM best layer by OOD TGT AUC:       {best_ood} "
          f"(in={results['per_layer'][str(best_ood)]['in_domain']['tgt_auc']:.3f}, "
          f"ood={results['per_layer'][str(best_ood)]['ood']['tgt_auc']:.3f})")

    # Side-by-side detail at the RFM-best-in-domain layer
    if have_lr:
        L = best_in
        lr_layer = lr_results["per_layer"].get(str(L))
        if lr_layer:
            rfm_in = results["per_layer"][str(L)]["in_domain"]
            rfm_ood = results["per_layer"][str(L)]["ood"]
            lr_in = lr_layer["linear"]["in_domain"]
            lr_ood = lr_layer["linear"]["ood"]
            print(f"\nLayer {L}: LR vs RFM head-to-head")
            print(f"  {'method':>6} {'domain':>10}  {'ID':>7} {'TGT':>7} {'MON':>7}")
            print(f"  {'LR':>6} {'in-domain':>10}  {lr_in['id_auc']:>7.3f} {lr_in['tgt_auc']:>7.3f} {lr_in['mon_auc']:>7.3f}")
            print(f"  {'LR':>6} {'OOD':>10}  {lr_ood['id_auc']:>7.3f} {lr_ood['tgt_auc']:>7.3f} {lr_ood['mon_auc']:>7.3f}")
            print(f"  {'RFM':>6} {'in-domain':>10}  {rfm_in['id_auc']:>7.3f} {rfm_in['tgt_auc']:>7.3f} {rfm_in['mon_auc']:>7.3f}")
            print(f"  {'RFM':>6} {'OOD':>10}  {rfm_ood['id_auc']:>7.3f} {rfm_ood['tgt_auc']:>7.3f} {rfm_ood['mon_auc']:>7.3f}")

    print("\nInterpretation hints:")
    print("  - If RFM OOD TGT >> LR OOD TGT: feature learning closes the OOD gap. Headline result.")
    print("  - If RFM OOD TGT ~ LR OOD TGT: feature learning doesn't help. The OOD gap is information-theoretic.")
    print("  - If RFM in-domain ~ LR but OOD higher: RFM finds a more transferable feature.")
    print("  - If RFM in-domain HIGHER than LR but OOD similar: RFM overfits to in-domain more.")


if __name__ == '__main__':
    main()


# # Try bw=1
# python step3_rfm.py --layers_subset -1 --bws 1 --regs 1e-3 --center_grads false

# # Try bw=100
# python step3_rfm.py --layers_subset -1 --bws 100 --regs 1e-3 --center_grads false

# # Try center_grads=true
# python step3_rfm.py --layers_subset -1 --bws 10 --regs 1e-3 --center_grads true

# # Try a different reg
# python step3_rfm.py --layers_subset -1 --bws 10 --regs 1e-2 --center_grads false



# python step3_rfm.py --bws 10 --regs 1e-3 --center_grads false
# python step3_rfm.py --bws 10 --regs 1e-3 --center_grads true

