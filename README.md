# CoFiE

<p align="center">
  <strong>CoFiE: Coarse-to-Fine Evidence Selection for Efficient Streaming Video Understanding</strong>
</p>

<p align="center">
  <a href="https://signalispupupu.github.io/cofie_proj_page/">Project Page</a> ·
  <a href="https://github.com/AstoriaJiang/CoFiE">GitHub</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/venue-EMNLP%202026%20Main-6f42c1" alt="EMNLP 2026 Main">
  <img src="https://img.shields.io/badge/framework-PyTorch-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/model-Qwen3--VL-1f6feb" alt="Qwen3-VL">
  <img src="https://img.shields.io/badge/status-research%20release-2ea44f" alt="Research release">
</p>

CoFiE is a research codebase for efficient streaming video understanding through coarse-to-fine evidence selection. The release contains Qwen3-VL evaluation code, frame-level redundancy reduction, visual-token dropping, attention-based ablations, and visualization utilities.

The core pipeline has two stages:

1. **Coarse frame selection** reduces redundant temporal observations before multimodal encoding.
2. **Fine visual-token selection** uses feature or attention signals to retain useful temporal evidence during generation.

This public tree contains code and configuration only. Model checkpoints, benchmark data, environments, training data, and generated results remain outside Git.

> **Current release.** The public entry points are the Qwen3-VL evaluation scripts. Full training and additional model ports are maintained separately.

## Contents

- [Repository structure](#repository-structure)
- [Environment setup](#environment-setup)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Reproducibility](#reproducibility)
- [License and attribution](#license-and-attribution)
- [Citation](#citation)

## Repository structure

~~~text
.
├── eval_qwen3vl/
│   ├── qwen3_vl/                         # Baseline Qwen3-VL implementation
│   ├── qwen3_vl_attn_v2/                 # Attention-based token dropping
│   ├── qwen3_vl_hdiff/                   # HDiff / DTD implementation
│   ├── qwen3_vl_hdiff_parellel/          # Main parallel evaluation path
│   ├── qwen3_vl_hdiff_parellel_attn_v3/ # Attention v3 experiments
│   ├── qwen3_vl_clip/                    # CLIP frame-selection baseline
│   └── scripts/                          # Benchmark and visualization entry points
├── scripts/                              # Dataset and result utilities
└── eval_qwen3vl/requirement.txt          # Evaluation dependencies
~~~

## Environment setup

The reference environment uses Python 3.10, CUDA, PyTorch 2.8, and Transformers 5.6.2.

~~~bash
git clone https://github.com/AstoriaJiang/CoFiE.git
cd CoFiE

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r eval_qwen3vl/requirement.txt
~~~

Provide a compatible Qwen3-VL checkpoint and benchmark data separately. The evaluation scripts accept local paths through environment variables or command-line arguments; no server-specific absolute paths are required.

## Evaluation

The main HDiff/DTD StreamingBench entry point is:

~~~bash
CKPT_PATH=/path/to/Qwen3-VL-8B-Instruct \
TASK_CSV=/path/to/StreamingBench/Real_Time_Visual_Understanding.csv \
VIDEO_DIR=/path/to/StreamingBench/Real-Time_Visual_Understanding \
RUN_NAME=qwen3vl_hdiff_streamingbench \
bash eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_streamingbench_hdiff_parallel.sh
~~~

Start with a small subset before a full run:

~~~bash
USE_SUBSET=1 \
SAMPLE_FRAC=0.01 \
CKPT_PATH=/path/to/Qwen3-VL-8B-Instruct \
TASK_CSV=/path/to/StreamingBench/Real_Time_Visual_Understanding.csv \
VIDEO_DIR=/path/to/StreamingBench/Real-Time_Visual_Understanding \
bash eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/eval_streamingbench_hdiff_parallel.sh
~~~

Other entry points are grouped by method:

~~~text
eval_qwen3vl/scripts/qwen3_vl/                  baseline multi-benchmark evaluation
eval_qwen3vl/scripts/qwen3_vl_attn_v2/          attention v2 ablations
eval_qwen3vl/scripts/qwen3_vl_fps/              sampling and FPS studies
eval_qwen3vl/scripts/qwen3_vl_hdiff/            HDiff evaluation
eval_qwen3vl/scripts/qwen3_vl_hdiff_parellel/   parallel HDiff evaluation
eval_qwen3vl/scripts/qwen3_vl_clip/             CLIP frame-selection baseline
~~~

Keep predictions, traces, logs, and summaries in a result directory outside Git. Dataset licenses and benchmark-specific terms apply independently.

## Visualization

The repository includes utilities for inspecting frame similarity, attention maps, dropped visual tokens, and StreamingBench relevance. Check each script's help output before running:

~~~bash
python eval_qwen3vl/scripts/visualize_streamingbench_similarity.py --help
python eval_qwen3vl/scripts/visualize_attention_map.py --help
python eval_qwen3vl/scripts/visualize_drop_token.py --help
~~~

## Reproducibility

For every reported result, record the commit hash, checkpoint revision, Python/CUDA/PyTorch/Transformers versions, dataset revision, sampling settings, token-drop configuration, latency, accuracy, and error counts.

The public repository contains code and configuration only. It does not claim a benchmark score without the exact data, checkpoint, hardware, and evaluation protocol.

## License and attribution

The source project describes the code, data, and checkpoints as research-use-only, and inherited files include third-party components with their own notices. This repository does not add a blanket license over all inherited modifications. Read [NOTICE.md](NOTICE.md) and [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) before redistribution.

The implementation builds on Qwen3-VL, Transformers, TimeChat-Online, VideoLLM-online, StreamingBench, and related open-source projects. Please preserve their notices and attribution requirements. Model weights and datasets are not redistributed here.

## Citation

Please fill in the final author list and publication metadata from the public paper record before publication.

~~~bibtex
@inproceedings{cofie2026,
  title     = {CoFiE: Coarse-to-Fine Evidence Selection for Efficient Streaming Video Understanding},
  booktitle = {EMNLP 2026 Main Conference},
  year      = {2026},
  note      = {Update with final author and publication metadata}
}
~~~

## Contact

Please use the GitHub issue tracker for implementation questions and reproducibility reports.
