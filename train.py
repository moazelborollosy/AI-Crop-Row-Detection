"""Train RowNet on the HRSD training split (CPU friendly).

    python train.py [--epochs 50] [--holdout-videos IMG_1603_...]

Frames that fall inside an HRDED evaluation-clip time window are removed from training, so
the HRDED clips stay a time-disjoint held-out set. Targets are rendered from the pattern
labels, which lets geometric augmentation (flip, crop) stay exact.
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from rowdet import data as D
from rowdet import geometry as G
from rowdet.fit import label_bounds
from rowdet.model import IN_H, IN_W, MEAN, OUT_H, OUT_W, STD, RowNet
from rowdet.pipeline import DEFAULT_CKPT

CACHE_W, CACHE_H = 480, 270
REG_W = torch.tensor([4.0, 4.0, 4.0, 1.0, 1.0])


def load_cache(samples):
    imgs = [cv2.cvtColor(cv2.resize(cv2.imread(str(s.path)), (CACHE_W, CACHE_H), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2RGB) for s in samples]
    return imgs, [G.canon(s.params) for s in samples]


def augment(img, p, rng):
    if rng.random() < 0.5:
        img, p = img[:, ::-1], G.flip(p)
    # exact crop augmentation: random zoom / shift
    sx, sy = rng.uniform(0.78, 1.0), rng.uniform(0.85, 1.0)
    x0, y0 = rng.uniform(0, 1 - sx), rng.uniform(0, 1 - sy)
    p = G.crop(p, x0, y0, sx, sy)
    a, b = int(round(x0 * CACHE_W)), int(round(y0 * CACHE_H))
    img = img[b:b + max(8, int(round(sy * CACHE_H))), a:a + max(8, int(round(sx * CACHE_W)))]
    img = cv2.resize(np.ascontiguousarray(img), (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # photometric: lighting, white balance, saturation, blur, noise
    img = img ** rng.uniform(0.7, 1.5)
    img = (img - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.15, 0.15)
    img = img * np.array([rng.uniform(0.9, 1.1) for _ in range(3)], dtype=np.float32)
    gray = img.mean(2, keepdims=True)
    img = gray + (img - gray) * rng.uniform(0.5, 1.4)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.blur(img, (k, 1) if rng.random() < 0.5 else (1, k))
    if rng.random() < 0.3:
        img = img + np.random.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0, 1), p


def to_tensors(img, p):
    x = (torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return x, torch.from_numpy(G.soft_mask(p, OUT_H, OUT_W)), torch.from_numpy(G.to_reg(p))


class Set(torch.utils.data.Dataset):
    def __init__(self, imgs, params, train):
        self.imgs, self.params, self.train, self.rng = imgs, params, train, random.Random(0)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        if self.train:
            img, p = augment(self.imgs[i], self.params[i], self.rng)
        else:
            img = cv2.resize(self.imgs[i], (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            p = self.params[i]
        return (*to_tensors(img, p), torch.from_numpy(np.asarray(p, dtype=np.float32)))


def loss_fn(seg, reg, mask, target):
    l_seg = F.binary_cross_entropy_with_logits(seg[:, 0], mask)
    l_reg = (F.smooth_l1_loss(reg, target, beta=0.05, reduction="none") * REG_W).mean()
    return l_seg + 2.0 * l_reg, l_seg, l_reg


def evaluate(net, loader):
    net.eval()
    bce, errs = [], []
    with torch.inference_mode():
        for x, mask, target, p in loader:
            seg, reg = net(x)
            bce.append(F.binary_cross_entropy_with_logits(seg[:, 0], mask).item() * len(x))
            errs += [G.row_error(G.from_reg(r.numpy()), q.numpy())[0] for r, q in zip(reg, p)]
    return sum(bce) / len(errs), float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=str(DEFAULT_CKPT))
    ap.add_argument("--holdout-videos", nargs="*", default=[], help="video ids excluded from training entirely")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--pseudo", action="store_true", help="add the model-labelled video frames written by pseudo_label.py")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    hrsd = D.load_hrsd()
    windows = D.hrded_windows(D.load_hrded())
    train, n_overlap = D.drop_overlap(hrsd["train"], windows)
    val, n_overlap_val = D.drop_overlap(hrsd["validation"], windows)
    n_held = sum(s.video in args.holdout_videos for s in train)
    train = [s for s in train if s.video not in args.holdout_videos]
    val = [s for s in val if s.video not in args.holdout_videos]
    n_labelled = len(train)
    if args.pseudo:
        # self-training: same leak rules as the labelled frames, plus distance to the HRSD held-out frames
        pseudo = D.drop_near(D.drop_overlap(D.load_pseudo(), windows)[0], hrsd["validation"] + hrsd["test"])[0]
        train = train + [s for s in pseudo if s.video not in args.holdout_videos]
        print(f"pseudo-labelled frames added: {len(train) - n_labelled}")
    per_video = {}
    for s in train:
        per_video[s.video] = per_video.get(s.video, 0) + 1
    print(f"train {len(train)} (dropped {n_overlap} inside HRDED clip windows, {n_held} held-out video) | val {len(val)} (dropped {n_overlap_val})")
    print("train frames per video:", per_video)

    bounds = label_bounds([G.to_fit(G.canon(s.params)) for s in train[:n_labelled]])  # from human labels only
    print("search bounds from training labels:", {k: tuple(round(v, 3) for v in b) for k, b in bounds.items()})

    tr = torch.utils.data.DataLoader(Set(*load_cache(train), True), batch_size=args.batch, shuffle=True, drop_last=True)
    va = torch.utils.data.DataLoader(Set(*load_cache(val), False), batch_size=32)

    net = RowNet(pretrained=not args.no_pretrained)
    enc = list(net.features.parameters())
    head = [p for n, p in net.named_parameters() if not n.startswith("features.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr * (0.3 if net.pretrained else 1.0)},
                             {"params": head, "lr": args.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                total_steps=args.epochs * len(tr), pct_start=0.1)
    best, history, t0 = 1e9, [], time.time()
    for ep in range(args.epochs):
        net.train()
        run = np.zeros(3)
        for x, mask, target, _ in tr:
            seg, reg = net(x)
            loss, l_seg, l_reg = loss_fn(seg, reg, mask, target)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += [loss.item(), l_seg.item(), l_reg.item()]
        run /= len(tr)
        v_bce, v_err = evaluate(net, va)
        history.append({"epoch": ep + 1, "train_loss": run[0], "val_bce": v_bce, "val_row_err": v_err})
        flag = ""
        if v_bce + v_err < best:
            best, flag = v_bce + v_err, " *"
            meta = {"epochs": args.epochs, "best_epoch": ep + 1, "n_train": len(train), "n_pseudo": len(train) - n_labelled, "n_val": len(val),
                    "dropped_overlap": n_overlap, "holdout_videos": list(args.holdout_videos),
                    "imagenet_pretrained": net.pretrained, "val_bce": v_bce, "val_row_err_regression": v_err,
                    "input": [IN_W, IN_H], "train_minutes": (time.time() - t0) / 60}
            torch.save({"model": net.state_dict(), "bounds": {k: list(v) for k, v in bounds.items()}, "meta": meta}, args.out)
        print(f"ep {ep + 1:3d} loss {run[0]:.4f} seg {run[1]:.4f} reg {run[2]:.4f} | val bce {v_bce:.4f} row-err {v_err:.4f}{flag} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    with open(str(args.out).replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
