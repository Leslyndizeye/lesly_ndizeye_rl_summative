# PharmaStock: Mission-Based Reinforcement Learning 🏥💊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## My Mission
**To use data and AI to help pharmacies and clinics predict which medicines might run out, so they can order in advance and make sure patients always find the medicines they need.**

## Project Overview
I am trying to solve the problem of **medicine stock-outs**. This project implements an AI Agent that acts as a Pharmacy Manager...

This project implements and compares four distinct RL algorithms (**PPO, DQN, A2C, REINFORCE**) to solve the stochastic inventory control problem.

---
## Video Recording: https://www.loom.com/share/97ff9e5be91a449ba5df50849eb3f606

## Features
* **Custom Gymnasium Environment:** A verified simulation of supply chain logistics with lead times and uncertain demand.
* **4 RL Algorithms:**
    * **PPO, DQN, A2C** (via Stable Baselines3).
    * **REINFORCE** (Custom implementation from scratch using PyTorch).
* **HD Visualization:** A custom Pygame renderer with animations, dynamic stock bars, and particle effects.
* **Hyperparameter Tuning:** Extensive experimentation with 10 different configurations per algorithm.
* **Automated Analysis:** Scripts to generate comparative plots and learning curves.

---

## 📂 Project Structure

```text
lesly_ndizeye_rl_summative/
├── environment/
│   ├── custom_env.py        # The Logic: State, Actions, Rewards
│   └── rendering.py         # The Visuals: Pygame HD Renderer
├── training/
│   ├── train_all.py         # Master script to train all models
│   ├── ppo_training.py      # PPO Implementation (Stable Baselines3)
│   ├── dqn_training.py      # DQN Implementation (Stable Baselines3)
│   ├── a2c_training.py      # A2C Implementation (Stable Baselines3)
│   └── reinforce_training.py# REINFORCE Implementation (Custom PyTorch)
├── models/                  # Saved trained models (.zip and .pth)
├── results/                 # Generated plots and performance logs
├── main.py                  # Entry point for Simulation/Video
├── generate_plots.py        # Script to create report diagrams
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
````

-----

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Leslyndizeye/lesly_ndizeye_rl_summative.git](https://github.com/Leslyndizeye/lesly_ndizeye_rl_summative.git)
    cd lesly_ndizeye_rl_summative
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## Usage

### 1\. Run the "Best Agent" Simulation (For Video)

This runs the pre-trained **PPO** model (the winner) with full HD visualization.

```bash
python main.py
```

*Flags:*

  * `--algorithm dqn` : Run the DQN agent.
  * `--algorithm reinforce` : Run the REINFORCE agent.
  * `--episodes 10` : Change number of test episodes.

### 2\. Train the Models

To retrain all agents from scratch (Warning: Takes time):

```bash
cd training
python train_all.py
```

*Or train individually:*

```bash
python training/reinforce_training.py
```

### 3\. Generate Report Plots

To generate the Bar Charts and Learning Curves used in the PDF report:

```bash
python generate_plots.py
```

-----

## Environment Details

### Observation Space (Normalized [0, 1])

The agent observes a vector of 4 values:

1.  **Current Stock:** Level of medicine on shelf.
2.  **Demand History:** Yesterday's patient count.
3.  **Pending Orders:** Stock arriving tomorrow.
4.  **Time:** Day of the month (0-30).

### Action Space (Discrete)

  * `0`: **Wait** (Order 0 units)
  * `1`: **Low** (Order 5 units)
  * `2`: **Medium** (Order 10 units)
  * `3`: **High** (Order 20 units)

### Reward Function

  * **+2.0**: Revenue per patient served.
  * **-10.0**: Penalty for Stockout (Critical).
  * **-0.1**: Holding Cost per unit.
  * **-2.0**: Fixed Ordering Cost.

-----

## Results Summary

| Algorithm | Best Config | Mean Reward | Stability | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **PPO** | Run 10 | **-7.00** | High | **WINNER: Most Reliable** |
| **DQN** | Run 10 | -12.00 | Medium | Good, but harder to tune. |
| **A2C** | Run 10 | -17.50 | Medium | Stable but lower returns. |
| **REINFORCE**| Run 10 | -53.10 | Low | High variance; risky. |

**Key Finding:** PPO demonstrated the best sample efficiency and stability, learning a "sawtooth" inventory policy that perfectly anticipates demand. REINFORCE suffered from high variance, occasionally crashing the supply chain.

-----

## 👤 

**Lesly Ndizeye**
