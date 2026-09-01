# Long-Horizon Dual-Arm Manipulation with π0.5

A real dual-arm robot completing an **18-subtask, ~4-minute packing task** end to end —
pick two boxes, fold the carton's side flaps, close the lid — driven by four fine-tuned
π0.5 policies with automatic subtask switching.

No human keypress during execution.

---

## Demo

> **[TODO: rollout GIF — full 4-stage run, 3× speed]**
> `assets/rollouts/full_chain_3x.gif`

| | |
|---|---|
| **[TODO] Success rollout** — complete 18-subtask run | **[TODO] Failure case** — wrong grasp order |
| `assets/rollouts/success_full.gif` | `assets/rollouts/failure_grasp_order.gif` |

---

## Key Results

| | |
|---|---|
| Task horizon | **18 subtasks**, 4 expert policies, ~4 min per full run |
| Real-robot chain | 4 stages run back-to-back, **zero manual intervention** |
| Subtask switching | **96 %** of held-out episodes complete all subtasks (70/73) |
| Switch timing error | median **−1 frame**; 76 % within ±10, 91 % within ±20 frames @15 Hz |
| Training data | 369 episodes / **236,957 frames** / 15 Hz / 3 cameras, all self-collected |
| Switcher latency | **36 ms** end-to-end on a laptop RTX 5060 (needs < 66.7 ms for 15 Hz) |

---

## Overview

### The task

A carton sits in the middle of the table. Boxes are scattered on both sides.
The robot must:

```
E0  right arm  →  grasp a box from the right area, place it in the carton     (2 subtasks)
E1  left arm   →  grasp a box from the left area, place it in the carton      (2 subtasks)
E2  both arms  →  brace and fold the two side flaps flat                      (7 subtasks)
E3  both arms  →  lift and close the left flap, then the right flap           (7 subtasks)
```

### Why it is hard

A single policy trained on the whole 4-minute demonstration does not work — the phases
have different contact modes, different active arms, and errors compound. The task is
split into four experts, which turns the problem into: *when do we hand over?*

### System

```
┌───────────────── laptop (robot control) ─────────────────┐   ┌─── GPU server ───┐
│  chain orchestrator  →  4 rollout clients                │   │  4 × π0.5 policy │
│         │                     │                          │   │     servers      │
│         │                     ├── websocket ─────────────┼──►│  (OpenPI serve)  │
│         │                     │                          │   └──────────────────┘
│         │                     ├── SocketCAN ──► dual arm │
│         │                     └── 3 × USB camera         │
│         └── subtask switcher (SigLIP2 + progress head, local GPU)
└──────────────────────────────────────────────────────────┘
```

---

## My Contributions

This project is **built upon** OpenPI / π0.5 and LeRobot (see
[Acknowledgements](#acknowledgements)). What I did:

**Task & data**
- Designed the 4-expert / 18-subtask decomposition of the long-horizon task
- Built the kinesthetic-teaching collection stack — task-config-driven state machine,
  per-frame subtask labelling with **zero manual second-pass annotation**
  (`src/collect/`, `configs/*.yaml`)
- Collected all 369 episodes / 236,957 frames myself on the real robot
- Failure-mode taxonomy and dedicated failure-case datasets for contrast experiments

**End-effector pose representation**
- Converted joint-space datasets to a 20-D absolute EE / 14-D relative action
  representation; wrote FK, damped-least-squares IK with null-space regularisation,
  and the rot6D/rotvec representation layer (`src/ee/`)
- **Chose the representation by measurement, not intuition**: rot6D for relative rotation
  hit `max|z| = 18.09` against a red line of 20; rotvec gave 12.00. Step-wise differencing
  gave 66.70 and was discarded in favour of chunk anchoring (`src/ee/probe_ee_repr.py`)
- Offline consistency check `joint → FK → relative → restore → IK → joint` as a hard gate
  before touching the real robot (`src/ee/replay_ee_ik.py`)

**Training pipeline**
- Fine-tuned four π0.5 LoRA policies (20k steps each) on the self-collected data
- Diagnosed and fixed a normalisation failure: an idle arm's action std collapses to
  ~1e-4 and z-scoring explodes — added a per-dimension std floor, asserted
  `max|z| < 20` on real batches in CI-style verification (`evaluation/verify_ee_dataset.py`)
- Gated training entry point that refuses to touch a GPU unless norm stats and pipeline
  verification both pass (`scripts/go_ee_train.sh`)

**Deployment on the real robot**
- 4-stage chain orchestrator with per-stage recording, shared run IDs, deferred video
  encoding, and inter-stage repositioning (`src/deploy/run_chain_4stage.py`)
- Joint-level safety: rate limiting + anti-windup clamping that bounds stall torque —
  added after diagnosing a J6 over-current trip caused by a 30° command-vs-actual gap
- Emergency stop that **holds position under power** rather than disabling — this
  hardware's brake does not hold, so a naive `disable()` drops the arm onto the workpiece
- Integrated a subtask progress head (SigLIP2 + MLP) for automatic switching, including a
  minimum-duration guard I added after finding the released config had **no lower bound**
  and could burn through a 140-frame subtask in 14 frames

**Evaluation & tooling**
- Held-out serial-replay evaluation of switch timing (error distribution, completion rate)
- Per-step diagnostic tracing so a bad switch can be explained instead of guessed at
- Rollout video compositor for reviewing runs (`visualization/make_run_overview.py`)

---

## Method

### Four experts, deterministic hand-off

Expert-to-expert switching uses a **deterministic counter**, not a classifier.

I trained a 3-way image classifier for this first: it reached 100 % per-frame accuracy but
its temporal profile was completely flat — at 90 % through an E0 episode it still predicted
E2 with 0 % confidence. It had learned *which dataset a frame came from*, not *when to
switch*. The datasets were recorded separately, so the decision boundary is not in the data.

The chain therefore advances experts by counting, and asks the progress head only
*"is this expert finished?"*.

### Subtask switching inside an expert

```
3 cameras → SigLIP2-so400m → MLP head → (progress, done, time-to-boundary)
                                              ↓ min = signal
   signal > τ for K=8 consecutive frames → wait D steps → advance subtask
```

Four mechanisms keep it stable:

| | |
|---|---|
| **Current frame only** | No frame index, no temporal model. If the network stalls, the input stops changing, so progress stops advancing — for free |
| **Stall gating** | `Δstate < 1e-5` freezes the counters, **but only while the signal is below τ** — a stall *after* the signal crosses means "done and holding", which should switch |
| **Hysteresis** | No re-switch within 10 steps |
| **Min / max duration** | Hard bounds on how early and how late a switch may fire |

The minimum-duration bound is mine. The released calibration had only an upper bound, and
its τ values were 1st-percentile estimates on the training distribution — some as small as
`1e-6`. On a real scene where the signal sits above τ from frame 1, a subtask would fire
after `K + D` ≈ 14 steps against a true length of 139. I set
`min_steps = 0.6 × p10(segment length)` per subtask, verified that every value falls below
the shortest observed segment, so it blocks false triggers without ever delaying a real one.

### Calling rate matters

The thresholds are all in **step** units, calibrated at 15 Hz. Calling the head at 5 Hz
drops switch accuracy from 76 % to 4 %. The switcher therefore runs on the **camera loop**,
not the policy inference loop, and is pipelined so the GPU round-trip stays off the
control path.

---

## Dataset

369 episodes, 236,957 frames, 15 Hz, three RGB cameras — all collected by kinesthetic
teaching on the real robot.

See **[`dataset/README.md`](dataset/README.md)** for the full schema, the 20-D/14-D EE
layout, subtask definitions, and per-column meanings.

Full data on Hugging Face — **link TBD**.

---

## Experiments

### Subtask switching, held-out set

Serial replay over 73 held-out episodes — errors accumulate across subtasks, which is the
closest offline proxy to real-robot behaviour.

| Expert | Episodes completed | Switches | Median error | ≤10 frames | ≤20 frames |
|---|---|---|---|---|---|
| E0 right-arm pick | 16/18 | 34 | −4 | 65 % | 91 % |
| E1 left-arm pick | 16/16 | 32 | +0 | 75 % | 84 % |
| E2 flap folding | 24/25 | 174 | −1 | 74 % | 92 % |
| E3 flap closing | 14/14 | 98 | +0 | 85 % | 90 % |
| **Total** | **70/73 = 96 %** | 341 | **−1** | **76 %** | **91 %** |

### Representation ablation

Measured on real data, normalised amplitude (red line = 20):

| Relative-rotation representation | `max|z|` | Verdict |
|---|---|---|
| rot6D | 18.09 | at the limit |
| **rotvec** | **12.00** | chosen |

| Action parameterisation | `max|z|` | Verdict |
|---|---|---|
| step-wise difference | 66.70 | unusable — 71 % of chunks have one arm still |
| **chunk-anchored to first frame** | within range | chosen |

### Real-robot latency budget @15 Hz (66.7 ms/step)

| Component | Measured |
|---|---|
| 3 × camera read | 0.6 ms |
| Switcher (SigLIP2 + head, RTX 5060) | 36 ms |
| Policy inference, amortised over a 50-step chunk | 12.4 ms |

Reducing the observation payload from full-resolution to the server's own 224×224
pre-resize cut inference round-trip from **620 ms to 222 ms** (2.77 MB → 0.45 MB) —
the client and server now run bit-identical resize, so model input is unchanged.

---

## Failure Cases

### Case 1 — Premature subtask switch

**Observation.** The left-arm expert advanced through both of its subtasks in under two
seconds; the arm never actually grasped the box.

**Diagnosis.** Instrumented every switch decision. The signal sat at 0.816 while τ was
0.001434 — **570× above threshold from the very first frame**, so `hits` incremented every
step and the switch fired at `K + D` = 14 steps against a true segment length of 139.
The released calibration had a `max_steps` upper bound but **no lower bound**.

**Fix.** Added `min_steps` per subtask, derived from training segment-length statistics.
Verified against the shortest observed segment for all 18 subtasks so it cannot delay a
legitimate switch. Post-fix, a real run produced 2/2/7/7 switches with per-subtask step
counts matching the training distribution (E0: 206/167 steps vs. training median 212/186).

**Open question.** On random-noise input the signal is far above τ, which means τ is
effectively an open gate. If real scenes behave the same way, switching degrades into a
fixed timer and τ should be **re-calibrated on real-robot data** rather than patched with
more guards. The per-step trace log was added to answer exactly this.

### Case 2 — J6 over-current trip

**Observation.** The left arm repeatedly froze at the same point in E2, joint indicator red.

**Diagnosis.** J6 was driving into its mechanical limit. The position servo's torque is
proportional to (command − actual); the command kept advancing while the arm was blocked,
so the gap grew to ~30° and current reached 9 A, tripping the joint.

**Fix.** Anti-windup clamping — the command may lead the measured position by at most
`stall_gap` radians, which bounds stall torque. I also verified that **null-space
regularisation cannot rescue this**: for a 7-DoF arm on a 6-DoF task the null space is
1-dimensional and its direction is not ours to choose.

### Case 3 — Missing camera in single-arm clients

**Observation.** `KeyError: 'left_wrist'` on E0/E1 once auto-switching was enabled.

**Diagnosis.** The progress head is trained with `n_view=3`, but single-arm clients
deliberately open only two cameras (the policy's repack uses two). Feeding zeros for the
third would be out-of-distribution input.

**Fix.** Open the third camera only when auto-switching is on, used by the switcher alone —
the policy's observation is unchanged.

---

## Code Structure

```
src/
├── deploy/            real-robot execution
│   ├── run_chain_4stage.py         ★ entry point: 4-stage orchestrator
│   ├── run_pi05_rollout_*_ee.py      per-expert rollout clients (record + execute)
│   ├── run_pi05_deploy_hezi_ee.py    dual-arm layer: arm control, IK, cameras, clamping
│   ├── rollout_boxpick_common.py     single-arm layer: keypad, device resolution
│   ├── switcher_{client,service}.py   subtask progress head (separate venv, IPC)
│   ├── goto_home.py / goto_start.py   closed-loop repositioning
│   └── traj_{teach,replay,replay_record}.py   kinesthetic trajectory record / replay
├── collect/           data collection state machines
└── ee/                joint ⇄ end-effector pose
    ├── fk_nero.py     forward kinematics (numpy only)
    ├── ik_nero.py     damped least squares + null-space
    ├── ee_repr.py     rot6D / rotvec, absolute ⇄ relative
    └── build_ee_dataset.py

configs/               task definitions (subtask prompts, cameras, failure taxonomy)
scripts/               training pipeline + policy-server launchers
evaluation/            dataset & pipeline verification (must ALL PASS before training)
visualization/         rollout video compositor
dataset/               format spec + sample episode
docs/                  design notes
assets/                demo GIFs and images
```

---

## Usage

> Requires the robot hardware, a CUDA GPU for the policy servers, and vendor SDK.
> Paths in configs are placeholders (`<DATA_ROOT>` etc.) — set them for your machine.

```bash
# 1. hardware check: CAN / web UI / driver
bash src/deploy/check_arms.sh

# 2. start the four policy servers (on the GPU machine)
GPU=0 PORT=8031 STEP=19999 bash scripts/serve/serve_rightbox_ee.sh
# ... one per expert

# 3. dry run — verifies orchestration without touching the robot
python3 src/deploy/run_chain_4stage.py --dry-run

# 4. real robot, read-only (inference + recording, no motion commands)
python3 src/deploy/run_chain_4stage.py --host <POLICY_SERVER_IP>

# 5. full execution with automatic subtask switching
python3 src/deploy/run_chain_4stage.py --host <POLICY_SERVER_IP> \
        --execute --auto-advance --auto-switch
```

Training a policy:

```bash
python3 src/ee/build_ee_dataset.py --src <joint-dataset> --dst <ee-dataset>
python3 src/ee/replay_ee_ik.py <ee-dataset>          # offline consistency gate
python3 evaluation/verify_ee_dataset.py <ee-dataset> --pipeline   # must ALL PASS
bash scripts/compute_norm_stats_ee.sh
python3 src/ee/floor_norm_stats_ee.py <norm_stats.json>
DRY=1 bash scripts/go_ee_train.sh                    # gate check, does not touch GPU
bash scripts/go_ee_train.sh <exp-name>
```

---

## Model Checkpoints

Weights are not in this repository. Training configuration:

| | |
|---|---|
| Base model | π0.5 (`pi05_base`), LoRA — `gemma_2b_lora` + `gemma_300m_lora` |
| Training steps | 20,000, checkpoints every 5,000 |
| Batch size | 32 (dual-arm) / 16 (single-arm) |
| LR schedule | cosine, warmup 1,000, peak 1e-4 → 1e-6 |
| Optimiser | AdamW, gradient clip 1.0 |
| Hardware | 1 × 32 GB / 48 GB GPU per run, ~10–21 h |

Best checkpoints → Hugging Face, **link TBD**. Intermediate checkpoints are not published.

---

## Acknowledgements

### Built upon

| Project | Used for |
|---|---|
| [OpenPI](https://github.com/Physical-Intelligence/openpi) / π0.5 | Base VLA model, training and serving infrastructure |
| [LeRobot](https://github.com/huggingface/lerobot) v2.1 | Dataset format, recording and video encoding |
| [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-224) | Vision encoder for the subtask progress head |

The four fine-tuned policies are LoRA adaptations of π0.5. Training and serving use
OpenPI's `scripts/train.py` and `scripts/serve_policy.py`; dataset I/O uses LeRobot.
**These frameworks are not my work.**

### Mine

Task decomposition, all real-robot data collection and its labelling pipeline, the
end-effector pose representation and its selection experiments, the joint-level safety
layer, the 4-stage chain orchestrator, subtask-switching integration and the
minimum-duration fix, evaluation tooling, and the failure analyses above.

---

## Status

Actively developed. Known open items are listed in [`docs/TODO.md`](docs/TODO.md).
