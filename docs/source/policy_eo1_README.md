# EO-1 (eo1)

This repository contains the LeRobot integration of **EO-1**, a Vision-Language-Action policy for robot control. The current LeRobot port focuses on robot-control training and evaluation through the standard LeRobot policy, dataset, processor, and environment interfaces.

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

## Training Example

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=eo1 \
  --policy.vlm_base=Qwen/Qwen2.5-VL-3B-Instruct \
  --policy.dtype=bfloat16 \
  --policy.attn_implementation=sdpa \
  --policy.gradient_checkpointing=false \
  --job_name=eo1_training \
  --steps=300000 \
  --batch_size=16 \
  --policy.device=cuda
```

For multi-GPU training, launch the same command with `accelerate launch` and tune `batch_size`, `num_workers`, and scheduler settings for your hardware.

---

## Processor Pipeline

1. Rename observation keys when a `rename_map` is provided.
2. Normalize visual, state, and action features using dataset statistics.
3. Build Qwen2.5-VL conversation messages with EO-1 state and action tokens.
4. Tokenize and batch text/image inputs with the Qwen2.5-VL processor.
5. Unnormalize predicted actions after sampling.

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
