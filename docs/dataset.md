# The EPIC Lab lower-limb biomechanics dataset

**Camargo, J., Ramanathan, A., Flanagan, W., & Young, A. (2021).** *A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions.* Journal of Biomechanics. https://doi.org/10.1016/j.jbiomech.2021.110320

Download: [EPIC Lab, Georgia Tech](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/)

This page documents the source data as it arrives, before anything in this project touches it. What this project *does* with it — the six-class mapping, the windowing, the exclusions — is in the notebook.

---

## 1. Overview

Biomechanical and wearable-sensor recordings of the lower limbs across several locomotion modes: level-ground walking, treadmill, stairs and ramps. The dataset is meant as a unified basis for locomotion-recognition systems and for the control of assistive robotic devices — exoskeletons and prostheses. It is organised by subject, date, mode, sensor and file.

## 2. Participants

**22 able-bodied adults.**

## 3. Sensors and available signals

Every file carries a `Header` column with the timestamp, which is what synchronisation is built on.

| Table | Contents | Rate |
|---|---|---|
| `conditions` | Experiment description, locomotion-mode labels, terrain conditions (speed, ramp inclination, step height) | 1000 Hz |
| `emg` | Electromyography from 11 muscles, band-pass filtered 20–400 Hz | 1000 Hz |
| `fp` | Ground reaction forces from force plates | 1000 Hz |
| `gcLeft` / `gcRight` | Gait-cycle segmentation from heel strike or toe off, per leg | 200 Hz |
| `gon` | Goniometers: hip (frontal, sagittal), knee (sagittal), ankle (frontal, sagittal) | 1000 Hz |
| `id` | Inverse dynamics, computed in OpenSim from ground reaction forces and motion-capture kinematics | 200 Hz |
| `ik` | Inverse kinematics, computed in OpenSim from motion capture | 200 Hz |
| `ik_offset` | Inverse kinematics with angles relative to the static pose | 200 Hz |
| `imu` | Inertial units (accelerometer + gyroscope) on trunk, thigh, shank and foot | 200 Hz |
| `jp` | Instantaneous joint power, from joint moment and angular velocity | 200 Hz |
| `markers` | Motion-capture marker trajectories | 200 Hz |

**This project reads `conditions`, `imu` and `gon` only.** EMG, force plates and motion capture are excluded by design: an exoskeleton cannot carry them.

> The sampling rate of every table is *measured* in the notebook rather than taken from the dataset README, which documents a different one for the labels. The labels were found to share the IMU time base at 200 Hz.

## 4. Locomotion modes

The recordings are split into five directories by environment or activity.

### 4.1 Level-ground

Walking on flat ground at three speeds, along circular or shuttle courses, clockwise (CW) and counter-clockwise (CCW). The `conditions` files segment each trial into millisecond-level states:

- `idle` / `stand` — standing still
- `stand-walk` — transition from standing to walking
- `walk` — steady straight-line walking
- `turn1` / `turn2` — turning and changing direction within the course
- `walk-stand` — deceleration and transition to a stop

### 4.2 Treadmill — *excluded from this project*

Continuous walking on a moving belt, over a wide speed range split into 8 trials per subject.

Each file contains **several speeds**: the belt follows a dynamic profile reaching 4 steady plateaus, each held for about 32–33 seconds (32,000 samples at 1000 Hz), separated by acceleration and deceleration transients. The speeds increase systematically by **0.05 m/s** from one trial to the next:

| Trial | Speeds (m/s) |
|---|---|
| 01 | 0.50, 1.30, 1.70, 0.90 |
| 02 | 0.55, 1.35, 1.75, 0.95 |
| 03 | 0.60, 1.40, 1.80, 1.00 |
| … | *(+0.05 per trial)* |
| 08 | 0.85, 1.65, 2.05, 1.25 |

Unlike level-ground, there are no turns and no frequent stop transitions, which makes these data ideal for isolating the pure biomechanics of locomotion at different speeds. This project excludes the condition by design.

### 4.3 Stairs

Ascent and descent of structured staircases, recorded at 4 different step heights (annotated `1`–`4`). The labels also specify the leading leg used on the first step, `l` or `r`.

As with level-ground, stair trials are whole time sequences rather than pure segments. Typical millisecond-level labelling:

- `idle` — standing before the obstacle or at the end of the trial
- `walk-stairascent` — transition from level walking to the first step up
- `stairascent` — the ascent proper
- `walk-stairdescent` — transition from level walking to the first step down
- `stairdescent` — the descent proper

### 4.4 Ramps

Ascent and descent of inclined flat surfaces, over 6 different inclinations (annotated `1`–`6`), with the leading leg again noted as `l` or `r`. The sequence structure matches the stairs:

- `idle`, `walk-rampascent`, `rampascent`, `walk-rampdescent`, `rampdescent`

### 4.5 Static

Markers and posture during a fully static phase, used to calibrate the biomechanical models (`ik_offset`) and body-segment lengths.

---

## 5. How this project maps the labels

Six classes: `idle`, `walk`, `stair_ascent`, `stair_descent`, `ramp_ascent`, `ramp_descent`.

Each transition is **absorbed into the mode it leads into** — `walk-stairascent` becomes `stair_ascent`. Every sample is additionally marked steady or transitional, which is what makes the error analysis in §4.3 of the notebook possible: it can score stationary and transition windows separately.

That convention has a cost, and the notebook measures it rather than hiding it. A window still labelled `idle` while a transition is under way is mostly movement, and `idle` collapses there to the worst precision and recall of any class in any condition — having been the strongest on stationary windows. The alternatives move that cost; they do not remove it.
