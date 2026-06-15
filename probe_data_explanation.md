# What data we use, and how — ViT and LLM

*Nonlinear-vs-linear activation-redundancy probe. Last updated 2026-06-15.*

**Goal:** test whether the redundancy in transformer activations is purely *linear*
(already captured by PCA / low-rank methods like FLAT-LLM, ASVD, MLA) or has a
*nonlinear* part those methods can't reach. We run it on one vision transformer
**and** one language model so the answer isn't modality-specific.

## The method is identical for both models

1. Take a **pretrained, frozen** model — we never train or fine-tune it.
2. Push data through it **once** and record the internal **activations** at 5 sites
   in each of 3 blocks: the MLP hidden layer, the attention output, and the
   **Q / K / V** projections. (These are exactly the places existing compression
   methods operate.)
3. Split those activation vectors into **train / val / test by source** (a whole
   image, or a whole text block — one source never spans two splits). On *train* we
   fit a rank-k PCA and a small nonlinear decoder; we early-stop on *val*; on
   held-out *test* we check whether the nonlinear decoder beats PCA at the same
   bottleneck size. A covariance-matched **Gaussian-null twin** runs alongside to
   subtract any overfitting floor.

The image/text data is just **realistic input to generate realistic activations** —
it is *not* training data.

## Side by side

| | **ViT side** | **LLM side** |
|---|---|---|
| Model (frozen) | DeiT-Small, ImageNet-pretrained | Llama-2-7b (fp16) |
| Data source | ImageNet **validation** images | WikiText-2 text |
| How much | **1,800 images** | first **358,400 tokens** (1,400 × 256-token blocks) |
| One "source" unit | 1 image → 197 patch tokens | 1 block → 256 tokens |
| Total activation rows | 354,600 | 358,400 |
| Split by source (70/15/15) | 1,260 / 270 / 270 images | 980 / 210 / 210 blocks |
| → train / val / test rows | ~248k / 53k / 53k | ~251k / 54k / 54k |
| Blocks probed | layers 1, 6, 9 | layers 5, 16, 26 |

## Two things this is NOT

- **Not model training.** Both models are frozen. "train/val/test" here only refers
  to splitting the *captured activations* to fit and test a small probe decoder.
- **Not WikiText's official train/test split.** Our split is by document at the
  activation level. (WikiText's official test set would only come into play later,
  if we measure the downstream perplexity of an actual compressor.)

## Reproducibility (verified paths)

- ViT data: `/work/hdd/bdjd/imagenet_10pct/val` (1000 classes, 5000 imgs; we sample 1,800)
- LLM data: **WikiText-2-raw-v1 `train`, loaded canonically in-script** via
  `datasets.load_dataset("wikitext","wikitext-2-raw-v1",split="train")` — no
  intermediate file, fully reproducible. (Cached at
  `/u/sesmat/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1`.) We then take
  the first 1,400 × 256-token blocks. The old non-reproducible `wikitext_train.txt`
  (a truncated 5 MB slice) is **no longer used** by this probe.
- Probe script: `/work/hdd/bdjd/hypergraph_pruning/probe_nonlinear3.py`
  (run as `python probe_nonlinear3.py <model> mlp,heads,q,k,v`)
