readme.txt — Task 1 (Random_Switch + Varying Maze)

1) Submission contents
- Task1_Gridworld_DynaQ.ipynb — main notebook (runs both experiments and creates plots)
- gridworld.py, config.py, plot.py — (NOT modified)
- requirements.txt — dependencies
- Task1_Report_Final.docx (or exported PDF) — report (≤ 5 pages including plots)

2) Setup (Python 3)
Install dependencies:
    pip install -r requirements.txt

3) How to run (Notebook)
Open and run all cells:
    jupyter notebook Task1_Gridworld_DynaQ.ipynb

The notebook runs:
  (A) Random_Switch averaged over 100 environments
  (B) Custom VaryingMazeEnv averaged over 100 environments

It produces:
- learning curve plots (*.png)
- zoom plots (*_zoom.png)
- curve values (*.csv)
- printed metrics: final mean return and last-50 mean return
- Demonstrations of CSV plots for both Random Switch and Varying Maze functionalities, showcasing the agent's performance.


## Custom Environment: VaryingMazeEnv (newly created)

I implemented a custom environment called VaryingMazeEnv that derives from GridworldEnv. The goal was to create a maze/labyrinth-like task that is
challenging for classical RL due to sparse rewards and misleading exploration paths, while still being “varying” (different layout per environment instance).

### Key properties
- Size: 12×12 grid
- Symbols:
  - 'S' = start
  - 'G' = goal (+100 terminal reward)
  - 'O' = water trap (−100 terminal reward)
  - '#' = wall (blocked)
  - ' ' = free cell
- Reward structure (true rewards):
  - Goal: +100
  - Water: −100
  - Otherwise: 0 per step (step cost is 0)

### How variation is created (different maze each run)
Each environment instance is generated from a random seed, producing a different maze layout. Variation comes from randomly placed branches, dead ends,
and traps, while keeping the overall structure maze-like (long corridors and many wrong turns).

### Solvable-by-construction (no BFS/DFS, no path search)
To ensure the environment remains solvable without performing any algorithmic path checking:
1) A guaranteed corridor from S to G is carved first (constructed path).
2) Random branches and dead ends are added off the corridor to increase difficulty.
3) Water traps are placed mainly at branch ends and in misleading side areas.
4) The corridor is never overwritten by obstacles, so at least one valid S→G path always exists.

This satisfies the requirement of a varying environment and avoids prohibited algorithmic analysis (no BFS/DFS).

### Proof it can be solved (plot)
The notebook/script produces a learning curve averaged over 100 different VaryingMazeEnv instances:
- Output plot: varying_maze_result.png (and a zoomed version if enabled)
- The curve shows the mean episode return rising close to +100, demonstrating that the environment can be solved by the agent.


4) Expected runtime (why it can take long)
This project can take significant time because the agent uses Dyna-Q: after each real environment step it performs many extra “planning” Q-updates,
implemented with Python loops and dictionary lookups.

Worst case estimate (if episodes often run close to max_steps):

Real environment steps:
- 100 envs × 1000 episodes × 600 steps = 60,000,000 env steps

Planning updates:
- if planning_steps = 150:
  60,000,000 × 150 = 9,000,000,000 Q-updates

These planning updates dominate runtime. In practice, episodes often terminate early (e.g., −100 deaths), so runtime is usually lower than worst case,
but still can be large.

Practical note: On a typical laptop/desktop CPU, the full “100 envs × 1000 episodes” run may take tens of minutes to >1 hour,
depending on CPU speed and planning_steps.
5) Parameters used (final evaluation settings)

5.1 Random_Switch (final run)
- env_name: random_switch
- num_envs: 100
- episodes: 1000
- max_steps: 600
- planning_steps: 100
- alpha: 0.12
- gamma: 0.99
- q_init: 2.0
- eps_start: 1.0
- eps_end: 0.01
- eps_decay: 0.992
- switch_safety: True
- avoid_fatal: True
- base_seed: 0

5.2 VaryingMazeEnv (final run)
- env_name: varying_maze
- num_envs: 100
- episodes: 1000
- max_steps: 600
- planning_steps: 100
- alpha: 0.12
- gamma: 0.99
- q_init: 2.0
- eps_start: 1.0
- eps_end: 0.02
- eps_decay: 0.993
- switch_safety: False   
- avoid_fatal: False     
- base_seed: 1000

6) Demo: RANDOM_SWITCH and VARYING_MAZE (short meeting)

During the short explanation meeting, I will demonstrate the project in two parts:

(A) Fast live run (1–5 environments)
I will run the same implementation with a reduced number of evaluations (num_envs = 1 to 5) so that it completes in seconds/minutes. 
This live run will demonstrate that the code executes correctly, the agent learns online, and the learning curve is produced.

Example demo configuration (typical):
- num_envs: 1–5
- episodes: 100–300
- max_steps: 200–300
- planning_steps: 20–50

(B) Showing the final required results (100 environments)
I will then show the final required performance by loading the precomputed CSV files from the full evaluation (num_envs=100, episodes=1000).
These CSV files contain episode, mean_return, and std_return and allow reproducing the final plots instantly without waiting several hours.

Files used:
- random_switch_result_curve.csv
- varying_maze_result_curve.csv

This approach ensures that I can present both a live execution of the algorithm and the final required 100-environment results within the limited meeting time.


7) Reproducibility / seeds
- The final plots average over 100 different environments (different seeds).
- If hyperparameter tuning was done, final evaluation should use a different seed range (do not reuse the same environments).

