# CARLA Autonomous Driving Research
![CARLA](https://img.shields.io/badge/CARLA-0.9.12-2C8EBB?style=for-the-badge)
![Unreal Engine](https://img.shields.io/badge/Unreal%20Engine-4.26-313131?style=for-the-badge&logo=unrealengine&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-FF6F00?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Model%20Training-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Perception-5C3EE8?style=for-the-badge)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-Decision%20Making-673AB7?style=for-the-badge)


![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=flat&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Compute-013243?style=flat&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C?style=flat)

This project implements  **End-to-End autonomous driving** in the CARLA high-fidelity simulator targeting real strategies in autonomous vehicle research:

- Lane keeping on a multi-lane highway (Town04)
- Smooth lane changes without jerky transitions on a multi-lane highway
- Arrow-guided navigation and turns — responding to road arrow markings to choose the correct lane
- Collision free driving with Obstacle detection/avoidance

Driving episodes are automatically recorded as timestamped **MP4s videos** via CARLA → OpenCV → ImageIO pipeline, enabling failure-mode analysis of behavior, robustness, across CNN, DQN, and PPO agents in closed-loop environments within the Carla environment.

## Table of Contents

- [Prerequisites](#Prerequisites)
- [CNN Imitation Learning](#Approach-1—CNN-Imitation-Learning)
- [Deep Q-Network (DQN)](#Deep_Q-Network_(DQN))
- [Proximal Policy Optimisation (PPO)](#Proximal_Policy_Optimisation_(PPO))
- [Evaluation Comparison](#evaluation-Comparison)
- [Future Work](#future-work)

### 📌 Prerequisites:

- Python 3.8
- CARLA 0.9.12
- Unreal Engine 4.26
- Linux
- GPU (recommended)- CUDA
- Conda 

# Approach 1 — CNN Imitation Learning

**Overview:**

With Supervised learning (Behavioral Cloning), a raw CNN learns to map RGB images to continuous control commands (steering, throttle, brake). A pure CNN predictions for speed fluctuate frame-to-frame as lighting and texture change, even when speed is constant. This hybrid architecture design(CNN + PID controller) separates perception from regulation: the CNN owns steering (requires scene understanding), a PID outputs a corrected throttle (requires only error feedback for stablility) eliminating speed fluctuations driving along the lane. This separation produces stable, smooth closed-loop driving.


The data was collected and trained in autopilot mode.(manual driving exhibited few divergence while driving which stimulated off-lane behaviour during evaluation)

**file** : `FINAL_CNNlane.ipynb`

**Dataset** : `final_datasept.csv`

**Training set**:`train.csv` 

**Test dataset** : `val.csv` 

**model path** : `new_best_model.pth`



**CNN Pipeline:**

<img width="820" height="319" alt="image" src="https://github.com/user-attachments/assets/e3b27d0f-38a3-4831-b2ca-ecbb388682fa" />

**Model Config:**

<table>
<tr>
<td width="50%" valign="top">

| **Data Preprocessing**            | **Details**                          |
|------------------------|-------------------------------------|
| **Sampling** | WeightedRandomSampler (>25 steering bins) |
| **Augmentations** | Brightness, gamma, blur, flip and translation for (+steer ) |
| **Normalization** | Pixel values → [-1, 1] per channel |

</td>
<td width="50%" valign="top">


## Training Configuration

| **Parameter**           | **Value**                  
|------------------------|---------------------------|
| **Optimiser**          | AdamW (lr=3e-4, wd=1e-5)  |  
| **Loss Function**      | Mean Squared Error (MSE)  | 
| **Mixed Precision**    | AMP + GradScaler          | 
| **Gradient Clipping**  | max\_norm = 5.0           | 
| **Epochs / Batch Size**| 20 / 16                   | 

</td>
</tr>
</table>

<p align="center">
  <img src="https://github.com/user-attachments/assets/a196c586-686d-460b-a6fb-9cd84307f6ec" width="400">
</p>


**Inference & Cons**

<table>
  <tr>
    <th>Inference</th>
    <th>Cons</th>
  </tr>
  <tr>
    <td>
      • Achieved closed loop lane keeping and shifting in multi-lane highways across multiple episodes covering 500-600m distance. <br>
      • PID speed regulation eliminated frame-by-frame neural throttle noise cleanly <br>
      • Lane change completion tracked with other evaluation metrics per-session with mean/max/std deviation statistics logged to CSV<br>
      •  With ego vehicle completing the laps  with stacked frames cnn inference and PID control maintaining the speed within the lane and augmentation strategies reducing overfitting. <br>
    </td>
    <td>
      • Covariate shift at curves and intersections — peak drift ±0.5 to 0.7m at curve entries vs ~0.31m on straights. These states have near-zero probability in the straight-road training distribution<br>
      • Silent failure mode — model predicts confidently wrong steer at novel states and PID tend to override <br>
      • The model works well in its training manifold and degrades gracefully at its edges. The principled fix is DAgger: running the current policy, collecting expert corrections at visited states, retrain - Dataset challenge.<br>
      • the model never sees off-nominal states like turns, traffic signals, arrowheads during training, so small errors compounding at deployment resulting in drifting off the lane from the current , resulting in collision and making it difficult during traffic-obeyed driving and shifting<br>
    </td>
  </tr>
</table>


# Approach 2 — Deep Q-Network (DQN)

The agent learns by trial-and-error, receiving rewards for lane-centre driving and penalties for collisions with no expert data needed. It is also extended with arrow signals states, enabling context-aware lane changes that CNN cannot achieve.

**file** : `FINAL_RNNlane.ipynb`

**file name** : `carla_rl_training.py`in FINAL_RNNlane.ipynb file

**model path** : `dqn_smooth/carla_dqn_smooth_final.pth`, `dqn_arrow_checkpoints_best_git/carla_dqn_arrow_best.pth`


## DQN Configuration & Reward Design

<table>
<tr>
<td valign="top" width="70%">

### DQN Configuration

| Component           | Value                              |
|--------------------|------------------------------------|
| Action Space       | 9 discrete (throttle, steer)       | 
| State (smooth)     | 15D vector                         | 
| State (arrow)      | +3 one-hot → 18D                   | 
| Replay Buffer      | 50K, batch 128                     | 
| Target Network     | Update every 10 episodes           | 
| ε-greedy           | 1.0 → 0.05          | 
| episodes           | 400 episodes          | 
</td>

<td valign="top" width="70%">

### Reward Shaping

| Signal                  | Reward |
|-------------------------|--------|
| Lane centre (<0.2m)     | +5.0   |
| Heading alignment       | +3.0   |
| Smoothness              | +2.0   |
| Correct arrow           | +10.0  |
| Successful lane change  | +20.0  |
| Collision               | −200   |
| Off-road                | −100   |
| Lane invasion           | −20    |
| Wrong arrow             | −15    |
| High jerk               | −5     |

</td>
</tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/d1fee155-df10-4ef5-8305-5261ec57bb5d" width="800"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ea077228-0672-4128-8b1a-29971384b214" width="400"/>
    </td>
  </tr>
</table>


**Inference & Cons**

<table>
  <tr>
    <th>Inference</th>
    <th>Cons</th>
  </tr>
  <tr>
    <td>
      • 15D state space + 9 discrete actions enables RL agent to learn basic lane keeping, obstacle avoidance <br>
      • Obstacle detection and avoidance within 50m range successfully triggered slow-down and braking states <br>
      • Dense reward shaping converge gave the agent enough signal to learn a stable lane-keeping policy reached consistent non-collision behaviour  <br>
      • Expert demonstrations proving RL framework can replace the need of labelled trained data  <br>
    </td>
    <td>
      • Agent is structurally blind to adjacent vehicles during manoeuvres - adding lateral radar sensors to state (+2 dims) be a fix <br>
      • Idle trap (reward hacking) per-tick - Agent found the reward function's global maximum with idle at centre +10.0 adding traffic signs made this worse <br>
      • Discrete jitter — Occational steer jumps between 0.0 to ±0.30 per tick produces visible yaw oscillation <br>
      • DQN is the right algorithm for **what** to do (change lane, brake, follow arrow). It is the wrong algorithm for **how** to do it smoothly. <br>
    </td>
  </tr>
</table>



# Approach 3 —  Proximal Policy Optimisation (PPO)

PPO with a Gaussian policy to produce continuous control actions, resulting in smoother trajectories and more stable driving compared to DQN. This is implemented across different maps like Town04, Town10 under different geographical conditions along with 20 different traffic spawned vehicles.

**file** : `FINAL_RNNlane.ipynb`

**file name** : `PPO_training in (FINAL_RNNlane.ipynb) `

**model path** : `PPO_best_git/ppo_lane_keeping_best.pth`

## PPO (Proximal Policy Optimization)

| Parameter            | Value                          |
|---------------------|--------------------------------|
| Episodes            | 1000 (500 + 500 checkpoint)    |
| Action Space        | Continuous 2D Gaussian         |
| State Space         | 12-dim vector                  |
| Max Steps / Episode | 2500                           |
| GAE                | 0.95                           |
| Discount          | 0.99                           |
| PPO Clip           | 0.2                            |
| Entropy Coefficient | 0.01                           |
| Epochs       | 10 per rollout                 |
| Optimizer           | AdamW (lr = 3e-4)              |

PPO reward uses **exponential decay** terms rather than flat threshold bonuses —  gives a smooth, continuous gradient signal at
every lateral offset and heading error, instead of a sharp step function.

```python
lane_reward    = 5.0 * exp(-3.0 * |lane_offset|)
heading_reward = 3.0 * exp(-2.0 * |heading_error|)
distance_reward = 0.1                                    # per tick, constant
speed_reward   = 2.0 * exp(-|speed - target| / 3.0)
steer_penalty  = -0.5 * |steering|
```
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/d8b359f6-764e-44e0-93ea-7e71479cc1c0" width="800"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ba9506cd-14b8-4b23-9881-ebe5acfc74d3" width="800"/>
    </td>
  </tr>
</table>


<img width="600" height="337" alt="clean_ep1_20260123_134708 mp4 (1)" src="https://github.com/user-attachments/assets/70d0536b-6ab2-41cd-bf1a-63debaf4400c" />


**Inference & Cons**

<table>
  <tr>
    <th>Inference</th>
    <th>Cons</th>
  </tr>
  <tr>
    <td>
      • 100% safety with 20 traffic vehicles on roads after 1000 episodes with 1500 m distance and 82% success rate  <br>
      • Obstacle detection and avoidance within 50m range successfully triggered slow-down and braking states <br>
      • Continuous Gaussian policy produced smooth, stable control — no discrete jitter <br>
      • Successful with different town maps(Town04, Town10) and different climate  <br>
    </td>
    <td>
      • Negligence to other traffic signs - since they are not trained with them <br>
      • Reward misspecification - Agent on learning slower is easier to stay centred with speed averaged to 10m/s- Speed reward weight needs increasing <br>
      • Rare Scenario Problem - curriculum scheduling to ensure the rare scenario is encountered often enough to learn <br>
    </td>
  </tr>
</table>


## Evaluation Comparison

| Metric               | CNN + PID              | DQN                     | PPO                          |
|---------------------|-----------------------|-------------------------|------------------------------|
| Cruise speed        | 25 km/h            | ~5–7 km/h         | 15-18 km/h              |
| Typical lane offset | 0.05–0.31 m           | ~ 0.2–0.4 m        | ~ 0.04–0.22 m                |
| Peak drift          | ±0.51–0.77 m          | -                       | -          |
| Max lane offset     | ~0.87 m               | ~0.5 m                        | 0.151 m            |
| Success rate        | Qualitative          | —                       | 100% (formal)                |
| Distance covered    | ~200 m      | ~ 300 m                      | 1050 m          |
| Traffic light aware |  No       | No                      | Yes                           |
| Arrow following     | No                    | Yes (attempts)     | yes                           |
| Speed control       | PID-regulated         | Discrete steps          | Gaussian continuous          |
| Obstacle avoidance | No | No | Slows/brakes  |
| Lane change on obstacle | No | No | yes |

## Running ( All files are in .ipynb format)

With Bash 

 **1. Start CARLA server (in a separate terminal):**  ./CarlaUE4.sh -quality-level=Low -fps=20 -RenderOffScreen

**2. Collect CNN training data (ipynb file) or create a data_collection.py file:**  python3 data_collection.py 

**3. Train CNN  (ipynb file) or create a train.py file:**  python3 train.py 

**4. Train DQN or PPO or create a train.py file:**  python3 filepath/train.py --episodes 300

**5. Evaluate DQN/PPO or create a carla_evaluation.py file:**  python3 carla_evaluation.py  --model_path/ppo_lane_keeping_final_continued.pth --episodes 1 --max_steps 2500



## Future Work
This work is the empirical foundation of my PhD research application in autonomous driving, broader robotics and deep RL. The planned research extensions are:

- **Safe RL** — Constrained Policy Optimisation (CPO) for collision avoidance as a hard constraint
- **World Models and Predictive Planning** – Integrate learned environment models to predict future vehicle and pedestrian behaviour for safer decision making.
- **Vision-Language Driving Agents** – Extend the perception module with vision-language models to enable natural-language instruction following (e.g., "park behind the truck").
- **Multi-Agent Driving Scenarios** – Study interactions with pedestrians and other vehicles using multi-agent reinforcement learning and social navigation techniques.
- **Sim-to-real transfer** — domain randomisation across weather, lighting, and road surface in CARLA to improve policy robustness
- **Explainability** — attention-based saliency maps to identify which scene regions drive control decisions
- **Robotic Manipulation Transfer** – Adapt the perception-learning-control pipeline to robotic arm manipulation tasks, exploring shared representations across driving and manipulation domains.

















