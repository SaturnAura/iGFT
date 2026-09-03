# Third-Party Components

This directory vendors the frameworks that iGFT builds upon. They are kept
as separate, self-contained checkouts so that each project keeps its own
dependencies, entry points and license.

| Directory | Upstream | Used for | License |
| --- | --- | --- | --- |
| `LLaMA-Factory/` | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (v0.7.1.dev0) | LLM SFT, RM, PPO and pseudo-query generation | Apache-2.0 ([LICENSE](LLaMA-Factory/LICENSE)) |
| `SPTAR/` | [zhiyuanpeng/SPTAR](https://github.com/zhiyuanpeng/SPTAR) + [ColBERT](https://github.com/stanford-futuredata/ColBERT) | BEIR data prep, DPR/BM25CE/ColBERT retrieval, evaluation | MIT (ColBERT license is bundled under `SPTAR/dense_retrieval/retriever/col_bert/LICENSE`); see upstream for SPTAR |

## Installation

LLaMA-Factory (Python 3.8+, used by `scripts/{sft,generate,train_reward_model,ppo}.sh`):

```bash
cd third_party/LLaMA-Factory
pip install -e .
```

SPTAR retrieval environment (Python 3.7, used for DPR/ColBERT data prep and evaluation):

```bash
cd third_party/SPTAR
conda env create -f dense_retrieval/environment.yml   # env name: py37
conda env create -f dense_retrieval/retriever/col_bert/col37bert.yml  # env name: col37bert
```

> SPTAR ships small patches for `beir` and `sentence-transformers` under
> `SPTAR/dense_retrieval/packages/`. Follow its inline instructions after
> creating the `py37` environment if you plan to run DPR/BM25CE training.

## Local modifications

iGFT adds only lightweight glue on top of these checkouts:

* `SPTAR/dense_retrieval/data_process.py` accepts `--dataset`, `--weak-num`,
  `--ratio` and `--train-num` arguments (defaults reproduce the original flow).
* LLaMA-Factory's `data/dataset_info.json` has the iGFT dataset entries appended.
* All iGFT training configs live in [`configs/llamafactory`](../configs/llamafactory),
  not inside the vendored checkout.
