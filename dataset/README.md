# Dataset

Four real-robot datasets, one per expert policy. Collected by **kinesthetic teaching**
(zero-force / leader mode) on a dual-arm Nero robot — no teleoperation device involved.

Full datasets live on Hugging Face (link TBD). This directory keeps only the format
spec and one sample episode so the schema can be inspected without downloading ~10 GB.

---

## Overview

| Expert | Dataset | Episodes | Frames | Segments | Subtasks |
|---|---|---|---|---|---|
| E0 right-arm pick&place | `nero_right_box_pick_ee_v1` | 100 | 41,034 | 200 | 2 |
| E1 left-arm pick&place | `nero_left_box_pick_ee_v1` | 100 | 38,087 | 200 | 2 |
| E2 flap folding (stage 3/4) | `nero_hezi_closing_ee_v1` | 100 | 87,649 | 700 | 7 |
| E3 flap closing (stage 5/6) | `nero_stage56_flap_closing_ee_v2` | 69 | 70,187 | 483 | 7 |
| **Total** | | **369** | **236,957** | **1,583** | **18** |

All at **15 Hz**, three RGB cameras, LeRobot v2.1 format.

`Segments` = number of labelled subtask spans. Segment boundaries are derived from the
per-frame `prompt_index` run-lengths — **no manual second-pass annotation**.

---

## Recording setup

```
Cameras   third_view   640×480 @15Hz   scene overview (fixed mount)
          left_wrist   640×480 @15Hz   mounted on left gripper
          right_wrist  640×480 @15Hz   mounted on right gripper

Arms      7-DoF × 2 + parallel grippers, SocketCAN @1 Mbps
Mode      kinesthetic teaching (zero-force), operator physically guides the arms
```

### Why `action ≡ state`

```
action_source = kinesthetic_current_feedback_as_same_frame_action_target
```

During kinesthetic collection there is **no separate command stream** sent to the
controller. The `action` column records the *same-frame measured state* as the
demonstrated target — it is **not** the next-frame state.

This has a direct consequence for training: when one arm is idle for a whole segment,
its action deltas are near-constant (measured std ~1e-4), and z-score normalisation
divides by that std. Without a floor on those dimensions the normalised values explode.
See `src/ee/floor_norm_stats_ee.py`.

---

## Two representations

The same episodes exist in two forms. They are **not** interchangeable — the state and
action semantics differ.

| | Joint-space | End-effector pose (used by the deployed policies) |
|---|---|---|
| `observation.state` | 16 (7 joints + gripper) × 2 arms | 20 (see layout below) |
| `action` | 16, delta joints | 14, relative pose |
| Extra columns | — | `observation.state_ee20`, `action_ee20` |

> **Quick check**: if a dataset has no `*_ee20` columns, it is the joint-space version.

### 20-D absolute EE state

```
[0:3]    left  position xyz          (m, world frame)
[3:6]    left  rotation matrix col 0
[6:9]    left  rotation matrix col 1
[9]      left  gripper width         (m)
[10:13]  right position xyz
[13:16]  right rotation matrix col 0
[16:19]  right rotation matrix col 1
[19]     right gripper width
```

`rot6D` = first **two columns** of the rotation matrix (Zhou et al. 2019 continuous
representation), recovered by Gram-Schmidt. Columns, not rows — consistent everywhere.

TCP = fingertip, 0.138 m along the flange +z. Both arms share **one world frame**,
which is required for a dual-arm task — otherwise left/right positions are not comparable.

### 14-D relative action (computed at train time, not stored)

```
[0:3]   left  Δposition = p_i − p_0     (world frame, chunk anchored to first frame)
[3:6]   left  rotvec(R_i @ R_0ᵀ)
[6]     left  gripper width (absolute)
[7:14]  right, same layout
```

Two choices here were **measured, not guessed** (see `src/ee/probe_ee_repr.py`):

- **rotvec instead of rot6D for relative rotation** — rot6D gave `max|z| = 18.09`,
  against a red line of 20; rotvec brought it to 12.00.
- **Chunk-anchored instead of step-wise differencing** — in this task 71 % of chunks have
  one arm completely still; step-wise differencing gave `max|z| = 66.70`, unusable.

---

## Per-frame columns

| Column | Type | Meaning |
|---|---|---|
| `observation.state` | float32[16] | measured joint angles + gripper widths |
| `action` | float32[16] | demonstrated target (≡ state, see above) |
| `observation.state_ee20` | float32[20] | absolute EE pose (EE datasets only) |
| `action_ee20` | float32[20] | EE target (EE datasets only) |
| `observation.images.third_view` | video | scene camera |
| `observation.images.left_wrist` | video | left wrist camera |
| `observation.images.right_wrist` | video | right wrist camera |
| `prompt_index` | int64 | which subtask this frame belongs to |
| `prompt_text` / `prompt_text_zh` | string | language instruction (EN / ZH) |
| `subtask_instance_id` | int64 | 1-based subtask index within the episode |
| `subtask_start` / `subtask_end` | int64 | segment boundary flags |
| `episode_success` | int64 | 1 success / 0 failure |
| `failure_type` | string | failure taxonomy, `none` on success |
| `active_arm` | string | which arm is executing |
| `event_code` / `event_name` | int64 / string | collection-time events |

EE rollout datasets additionally carry `ik_pos_err`, `ik_ok`, `idle_left`, `idle_right`.

---

## Subtask definitions

### E0 right-arm pick & place (2)

```
1  Use the right arm to grasp a box from the right-side area.
2  Use the right arm to place the grasped box into an empty spot of the middle packing carton.
```

E1 is the mirror image (left arm, left-side area).

### E2 flap folding, stage 3/4 (7)

```
31  Move the right arm toward the lower part of the box's right side flap and brace the middle.
32  Move the left arm toward the lower part of the box's left side flap and brace the middle.
33  Use both arms at the same time to fold the left and right side flaps inward, then press flat.
41  Move both arms away from the carton flaps and withdraw to a safe position.
42  Move the right arm toward the front part of the box's right lower flap and gently push.
43  Move the left arm under the open flap, brace the left edge of the lower flap and push.
44  Move the right arm under the open flap, brace the left edge of the lower flap and push.
```

> ⚠ In `tasks.jsonl` the `task_index` is **not** the execution order for this dataset —
> execution order maps to `task_index [0,6,5,2,1,4,3]`. Deployment must apply that
> mapping; reading `task_index` as "which segment" silently loads the wrong prompt.

### E3 flap closing, stage 5/6 (7)

```
51  Raise the right arm, move it over the flaps, close to the carton top surface.
52  Open the right gripper and lower it so the two fingers brace the front and rear flaps.
53  Use the left arm to lift the left flap to near vertical, right arm keeps bracing.
54  Use the left arm to fold the left flap inward and close it, right arm keeps bracing.
55  Retract the right arm from above the carton and clear the flap area.
61  Use the right arm to lift the right flap from the right side to near vertical.
62  Use the right arm to fold the right flap inward and close it.
```

Here the dataset carries per-frame prompts directly — no index mapping needed.

---

## Files

```
dataset/
├── README.md              this file
├── dataset_format.md      LeRobot v2.1 on-disk layout + how to read it
├── sample_episode/        one episode, all columns, low-res video   [TODO]
└── visualize_episode.py   plot joint / EE trajectories, dump frames
```

## Download

Full datasets: **Hugging Face — link TBD**

```python
# once published
from datasets import load_dataset
ds = load_dataset("<hf-user>/nero-hezi-closing-ee-v1")
```
