# EO-1 (eo1)

This repository contains the LeRobot integration of **EO-1**, a Vision-Language-Action policy for robot control. The current LeRobot port focuses on robot-control training and evaluation through the standard LeRobot policy, dataset, processor, and environment interfaces.

EO-1 combines a Qwen2.5-VL vision-language backbone with EO-1 state/action special tokens and a continuous flow-matching action head. The policy consumes camera observations, a language task, and proprioceptive state, then predicts a chunk of continuous robot actions.

---

## Model Overview

| Component             | EO-1 in LeRobot                                                                   |
| --------------------- | --------------------------------------------------------------------------------- |
| Policy type           | `eo1`                                                                             |
| Backbone              | Qwen2.5-VL, configured by `policy.vlm_base`                                       |
| Action representation | Continuous action chunks sampled with flow matching                               |
| Default horizon       | `chunk_size=8`, `n_action_steps=8`                                                |
| State/action padding  | `max_state_dim=32`, `max_action_dim=32`                                           |
| Inputs                | One or more visual observations, `observation.state`, and `task` text             |
| Normalization         | Visual identity normalization; state/action mean-std normalization by default     |
| LeRobot scope         | Robot-control finetuning, checkpoint loading, and evaluation through LeRobot CLIs |

The processor pipeline formats each sample as a Qwen2.5-VL conversation. Images are inserted as visual content, the state is represented by EO-1 state placeholder tokens, and the action span is represented by EO-1 action placeholder tokens. During training, the model learns to denoise continuous action chunks at those action-token positions.

---

## Installation

After installing LeRobot, install EO-1 dependencies with:

```bash
pip install -e ".[eo1]"
```

For LIBERO evaluation or training, install the LIBERO extra as well:

```bash
pip install -e ".[eo1,libero]"
```

EO-1 can request different Qwen attention backends through `policy.attn_implementation`. Use `sdpa` as the portable default. Use `flash_attention_2` only in environments where `flash_attn` is installed and compatible with your CUDA/PyTorch stack.

---

## Data Requirements

EO-1 expects a LeRobot dataset with:

- At least one visual observation feature, such as `observation.images.image`
- A proprioceptive state feature named `observation.state`
- An action feature named `action`
- Task text available through the standard LeRobot `task` field

If your dataset uses different camera names, pass a `rename_map` to map your dataset keys to the names used by the policy or evaluation environment.

---

## Usage

To create a new EO-1 policy from a Qwen2.5-VL backbone, set:

```bash
policy.type=eo1
```

You can also override the backbone source:

```bash
policy.vlm_base=Qwen/Qwen2.5-VL-3B-Instruct
```

Once you have a LeRobot-format EO-1 checkpoint, load it in the standard way:

```bash
policy.path=your-org/your-eo1-checkpoint
```

---

## Training Example

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=eo1 \
  --policy.vlm_base=Qwen/Qwen2.5-VL-3B-Instruct \
  --policy.dtype=bfloat16 \
  --policy.attn_implementation=sdpa \
  --policy.gradient_checkpointing=true \
  --output_dir=./outputs/eo1_training \
  --job_name=eo1_training \
  --steps=3000 \
  --batch_size=4 \
  --policy.device=cuda
```

For multi-GPU training, launch the same command with `accelerate launch` and tune `batch_size`, `num_workers`, and scheduler settings for your hardware.

---

## Key Configuration Options

| Parameter                              | Default                       | Description                                                         |
| -------------------------------------- | ----------------------------- | ------------------------------------------------------------------- |
| `vlm_base`                             | `Qwen/Qwen2.5-VL-3B-Instruct` | Backbone used to initialize a new EO-1 policy                       |
| `dtype`                                | `auto`                        | Backbone dtype request: `auto`, `bfloat16`, or `float32`            |
| `attn_implementation`                  | `None`                        | Optional Qwen attention backend, for example `sdpa`                 |
| `gradient_checkpointing`               | `false`                       | Reduces memory usage by checkpointing Qwen/flow-head computations   |
| `chunk_size`                           | `8`                           | Number of future actions predicted per policy call                  |
| `n_action_steps`                       | `8`                           | Number of actions consumed before sampling a new chunk              |
| `num_denoise_steps`                    | `10`                          | Number of flow-matching denoising steps during action sampling      |
| `max_state_dim`                        | `32`                          | State padding dimension for the EO-1 flow head                      |
| `max_action_dim`                       | `32`                          | Action padding dimension for the EO-1 flow head                     |
| `force_fp32_autocast`                  | `true`                        | Keeps flow-head computations in fp32 under global autocast          |
| `supervise_padding_action_dims`        | `true`                        | Whether padded action dimensions contribute to the training target  |
| `supervise_padding_actions`            | `true`                        | Whether padded future action rows contribute to the training target |
| `image_min_pixels`, `image_max_pixels` | Qwen-style pixel bounds       | Controls Qwen2.5-VL image resizing in the EO-1 processor            |

---

## Processor Pipeline

EO-1 registers its own pre/post processors through `make_eo1_pre_post_processors`:

1. Rename observation keys when a `rename_map` is provided.
2. Normalize visual, state, and action features using dataset statistics.
3. Build Qwen2.5-VL conversation messages with EO-1 state and action tokens.
4. Tokenize and batch text/image inputs with the Qwen2.5-VL processor.
5. Unnormalize predicted actions after sampling.

This keeps EO-1 compatible with the same LeRobot training and evaluation entrypoints used by other policies.

---

## Current Status

The LeRobot integration currently covers the robot-control path: supervised finetuning, action sampling, checkpoint save/load, processor serialization, and LIBERO-style evaluation. The full EO-1 interleaved pretraining and multimodal text-generation workflows live in the upstream EO-1 project and are not part of this LeRobot policy README yet.

Open items to fill in before final publication:

- Public LeRobot-format EO-1 checkpoint names
- Reproducible benchmark commands and final numbers
- Architecture figure hosted on the Hugging Face documentation image dataset
- Hardware-specific deployment examples

---

## Citation

If you use EO-1, please cite:

```bibtex
@article{eo1,
  title={EO-1: Interleaved Vision-Text-Action Pretraining for General Robot Control},
  author={Delin Qu and Haoming Song and Qizhi Chen and Zhaoqing Chen and Xianqiang Gao and Xinyi Ye and Qi Lv and Modi Shi and Guanghui Ren and Cheng Ruan and Maoqing Yao and Haoran Yang and Jiacheng Bao and Bin Zhao and Dong Wang},
  journal={arXiv preprint},
  year={2025},
  url={https://arxiv.org/abs/2508.21112}
}
```

---

## License

This LeRobot port follows the **Apache 2.0 License** used by LeRobot. Check the upstream EO-1 model and dataset pages for the licenses of released EO-1 checkpoints and data.
