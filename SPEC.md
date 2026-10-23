# SWARM-LINK — Project Specification

**Project:** Learned Swarm Coordination for Transmitter Localization under Connectivity Constraints (Project 5)  
**Student:** Fausto Zamparelli  
**Purpose:** Single technical reference for the project and for AI coding assistants.

> **How to use this file:** The *Project requirements* section describes the established project scope. *Implementation guidance* describes suggestions that can be used immediately but are **not mandatory design decisions**. *Open decisions* are deliberately unresolved: propose and discuss solutions rather than treating arbitrary choices as official requirements. Scientific papers and software documentation are references, not imposed architectures.

## 1. Objective

Develop a **multi-agent reinforcement learning (MARL)** strategy that coordinates multiple drones to **localize an unknown radio transmitter** using **direction-of-arrival (DoA) measurements**, while keeping the swarm connected through drone-to-drone and/or drone-to-ground-station radio links.

Two objectives compete:

- **Accurate localization:** Drone placement and movement should provide informative measurements from different positions.
- **Communication connectivity:** Every drone must be able to reach the ground station, either directly or via other drones acting as relays. Wider separations may improve localization geometry but make communication more difficult.

The principal research question is whether **learned swarm coordination outperforms fixed drone formations** under connectivity constraints, and how localization accuracy trades off against those constraints.

This is a **simulation and algorithm evaluation project**. A real-world UAV deployment is not an established deliverable.

## 2. Project requirements

The official scope comprises all six tasks:

1. **Swarm and sensing model.** Define multiple drones with DoA measurements, plus a model of drone-to-drone and drone-to-ground-station communication range.
2. **Fixed-formation baselines.** Evaluate localization accuracy for different numbers of drones and formation geometries.
3. **Multi-agent RL formulation.** Define the learning environment, including agent states/observations, actions and a reward combining localization accuracy and connectivity.
4. **Learned coordination.** Train a coordination strategy and compare its results with the fixed-formation baselines.
5. **Accuracy–connectivity trade-off.** Quantify performance for different swarm sizes and connectivity requirements.
6. **Generalization.** Test the learned strategy on transmitter locations, swarm sizes or link conditions that were not used during training.

**Required outcomes:**

- A working multi-agent RL environment and **trained coordination policy**.
- A quantitative comparison of learned and fixed formations across swarm sizes.
- A quantitative localization-accuracy versus connectivity trade-off.
- A generalization assessment on unseen conditions.
- Reproducible simulation code, results, documentation and a final project report.

A fixed-formation simulator alone is **not** the final deliverable; learned coordination and comparative experiments remain essential.

## 3. Implementation guidance already provided

These are **recommended starting conditions**, not fixed parameters or mandatory methods:

### Simulation and data

- **Begin in 2D** for simplicity; move toward 3D only later if justified.
- **Generate a synthetic dataset** rather than assuming an existing Project 5 dataset. The implementation must define transmitter placements, measurement noise and communication assumptions.
- A simple example noise model is **Gaussian noise on measured DoA angles**; the distribution and noise level are not specified requirements.
- A simple example communication model connects two drones if their distance is below a configurable threshold; the numerical threshold is unspecified.
- Example 3D terrains/scenes and generation scripts may be made available by the researchers, but their availability should **not** be assumed.

### Knowledge, sensing and estimation

- In the initial setting, drones **may be assumed time-synchronized** and to **know each other's positions**.
- An individual drone is intended to use **its own position, its own measurements and information available from connected drones**. The precise observation/message representation is not prescribed.
- Use DoA measurements from multiple drones to estimate the transmitter location. **Least-squares triangulation** is a suggested initial method, not a required final estimator.
- Evaluate localization error over **multiple transmitter positions**.
- The **Cramér–Rao lower bound (CRLB)** is recommended as a theoretical accuracy reference; its use as a training objective is not specified.
- Check whether every drone can reach the **ground station directly or via multi-hop connections**. Being connected only to another drone is insufficient if the network cannot reach the station.

### Progression to learning

- Start by evaluating **fixed baseline formations**.
- Then introduce **sequential decisions** in which drones move, take new measurements and update the transmitter-position estimate.
- The **number of time steps** is not specified; it should be determined after studying initial simulation results and convergence.
- An optional harder extension involves drones that are not time-synchronized and/or do not know other drones' positions, requiring synchronization and relative localization before the mission.

### Technology

The researchers identified **Python, NumPy, SciPy, PyTorch or Stable-Baselines3**, and the broader simulation tools **Sionna, Ansys and AirSim** as relevant technologies. **None was imposed as a mandatory dependency for the initial 2D simulator**, and no exact MARL library or algorithm was selected.

The initial localization/connectivity/MARL problem does **not require a wind simulation, 3D scene, Sionna ray tracing or a fully integrated digital twin** to begin. Those may become future integration work, but are not part of the stated Project 5 baseline requirements.

## 4. System concepts that must remain distinct

- **Transmitter:** The radio source whose position the drones are trying to estimate. The simulator may know ground truth for measurement generation and evaluation; the acting policies must not simply be given the hidden answer.
- **Drone:** A mobile agent that has a position, obtains DoA observations, can communicate within network limitations and may select movements.
- **Ground station:** The reference endpoint to which all drones must maintain direct or multi-hop reachability. It is not automatically the unknown transmitter.
- **DoA sensing:** Direction/bearing information, not a direct distance measurement, transmitter coordinate or received-power coverage map.
- **Localization estimator:** Converts multiple noisy observations into a source-position estimate.
- **Connectivity model:** Determines valid communication links and whether each drone has a path to the ground station.
- **Coordination policy:** Chooses drone configurations/movements to improve localization while respecting connectivity limitations.
- **Evaluator:** Uses simulator truth to compute localization error, connectivity outcomes and generalization performance, independently of what observations the policies receive.

The mathematical conventions (coordinates, angles, time, units, uncertainty and degeneracy handling) still need to be defined during implementation; this document does not silently choose them.

## 5. Researchers and who to contact

| Person | Role and relevant expertise | Best contact for |
|---|---|---|
| **Luca Barletta** | Politecnico di Milano academic supervisor; information theory and communication theory. Responsible for academic supervision and evaluation. | Project scope, deliverables, grading, communication/information-theoretic questions. |
| **Stefano Rini** | Research lead for SWARM-LINK; distributed systems/learning, machine learning, communications, control and communication-constrained autonomous drones. | SWARM-LINK problem formulation, coordination and RL research direction, integration priorities. |
| **Alessia Radicula** | Researcher working on RF propagation, Sionna simulation, coverage maps, radio-source localization and triangulation. | DoA and propagation assumptions, localization, uncertainty and RF-simulation integration. |
| **Arthur Garbuz** | Researcher with aerospace/aerodynamics expertise, working on wind/CFD, UAV trajectory planning and simulation integration. | Drone flight costs, wind effects, geometry/terrain simulation assets and connections to flight planning. |

**Primary SWARM-LINK supervision:** Luca Barletta and Stefano Rini. Alessia and Arthur are relevant technical research contacts, not confirmed replacements for those supervisors.

## 6. Related projects — potential interfaces only

- **Project 4: Neural Prediction of RF Coverage Maps (AIRWISE).** Produces synthetic scenes and ray-traced radio coverage maps, then trains a neural model to predict maps more quickly. **Potential collaboration:** share scene/ground-station coordinates or incorporate coverage maps into a more realistic connectivity model. **Important:** RF coverage strength maps do not automatically provide DoA observations or solve transmitter localization. **Contact:** Alessia.
- **AIRWISE wind/CFD projects (Projects 1–3).** Investigate geometric simplification, meshing and turbine wakes for drone flight simulation. **Potential collaboration:** consume future wind/flight-cost or obstacle/flyability fields as additional motion constraints. These are **not prerequisites** for the initial Project 5 simulator. **Contact:** Arthur.

No common scene format, API, shared dataset or integration deadline has been fixed.

## 7. Research literature (verified; not project requirements)

**No paper or algorithm below is mandatory.** Read by relevance, not by date. The key gaps are (1) DoA/Fisher-information geometry, (2) multi-hop connectivity to the ground station, and (3) learned multi-agent movement under those constraints. These papers overlap with parts of that problem, but **none establishes the exact SWARM-LINK specification**. Follow their actual observation models and assumptions; do not silently copy them.

### Core references

1. **[Top priority — closest classical predecessor] [Algorithms for Cooperative Active Localization of Static Targets With Mobile Bearing Sensors Under Communication Constraints](https://doi.org/10.1109/TRO.2015.2432612)** — Vander Hook, Tokekar & Isler, *IEEE Transactions on Robotics*, 2015. [Author-hosted full PDF](https://josh.vanderhook.info/media/pdf/tro2015cooploc.pdf). Mobile bearing sensors actively localize an unknown stationary radio target under distance-based inter-robot communication restrictions. Study the estimation-information criterion, online measurement-site planning and classical non-RL strategies. **Mismatch:** the paper permits temporary disconnection followed by rendezvous, rather than enforcing permanent multi-hop reachability to a ground station. An excellent inspiration for a stronger non-learning baseline, not a drop-in implementation.
2. **[Very high priority — professor-shared reference] [UAV Path Planning for Radio Frequency Signal Localization via CRLB-Based Uncertainty Minimization](https://doi.org/10.1109/ICASSP55912.2026.11463984)** — Chen et al., *ICASSP*, 2026. Relates RF propagation, localization uncertainty, CRLB, a Gaussian-process information surrogate and iterative route optimization. **Mismatch:** not a connectivity-constrained MARL framework. The IEEE abstract and metadata are verified; full paper details may require institutional access.
3. **[Very high priority — bearing geometry] [Optimal Geometry and Motion Coordination for Multisensor Target Tracking with Bearings-Only Measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385148/)** — Wang et al., *Sensors*, 2023. Derives Fisher-information/D-optimal bearing sensor configurations and coordinated movement. **Mismatch:** assumptions include favorable/equal sensor–target ranges and no SWARM-LINK ground-station connectivity condition; do not assume its geometry is optimal for constrained scenarios.
4. **[High priority — fixed-formation/CRLB research] [Robust Multi-UAV Placement Optimization for AOA-Based Cooperative Localization](https://doi.org/10.1109/TIV.2024.3362818)** — Zhou et al., *IEEE Transactions on Intelligent Vehicles*, 2024. Multi-UAV AoA localization, sensor-distance-dependent uncertainty, CRLB/SPEB and robust placement when source location is uncertain. **Mismatch:** more detailed antenna-array and robust-optimization assumptions, without the stipulated graph-to-ground-station constraint or MARL.
5. **[High priority — constrained RF-source RL] [Autonomous Tracking Using a Swarm of UAVs: A Constrained Multi-Agent Reinforcement Learning Approach](https://doi.org/10.1109/TVT.2020.3023733)** — Chen, Chang & Zhang, *IEEE Transactions on Vehicular Technology*, 2020. Cooperative RF-target localization/tracking formulated as a constrained Markov decision process with multi-agent RL and uncertainty-aware feedback. **Mismatch:** uses RSS sensing and a mobile target, not DoA bearings to a stationary transmitter; its constraints must not be equated automatically with SWARM-LINK's ground-station connectivity rule.
6. **[High priority — closest modern RF MARL] [Hierarchical Role-Based Multi-Agent Reinforcement Learning for UHF Radiation Source Localization with Heterogeneous UAV Swarms](https://doi.org/10.3390/drones10010054)** — Sun et al., *Drones*, 2026. Explicitly combines AoA/RSS transmitter localization, coordination policies, sensing geometry and constrained cluster communications in a 2D simulation. Useful for observation/reward designs and evaluation. **Mismatch:** relies on heterogeneous cluster-head/monitor roles and combined AoA+RSS, neither mandated for SWARM-LINK. Its architecture is a research example, not a prescribed solution.
7. **[High priority — connectivity-aware MARL] [Multi-Agent Deep Reinforcement Learning for Persistent Monitoring With Sensing, Communication, and Localization Constraints](https://doi.org/10.1109/TASE.2024.3385412)** — Mishra et al., *IEEE Transactions on Automation Science and Engineering*, online 2024 / volume 2025. Proposes graph-based GALOPP with agents reaching anchors directly or via peers, and compares constrained versus unconstrained non-RL baselines. [Author preprint](https://arxiv.org/abs/2109.06831). **Mismatch:** the monitored quantity is environmental coverage and the localization constraint concerns robots' own positions, not localization of an RF transmitter. Highly relevant to the *network/topology and RL* side.

### Optional follow-up references

- **[Multi-UAV MARL implementation] [Multi-UAV Cooperative Searching and Tracking for Moving Targets Based on Multi-Agent Reinforcement Learning](https://doi.org/10.3390/app132111905)** — *Applied Sciences*, 2023. A MAPPO-based search/tracking study including local information, neighbor exchange and fusion. Useful when designing a MARL simulator, but not an RF DoA localization model.
- **[AoA-only MARL] [Hybrid Reinforcement Learning Control for UAV Territory Guarding with Cooperative Angle-of-Arrival Localization](https://doi.org/10.1007/s11431-025-3331-4)** — *Science China Technological Sciences*, 2026. Coupling passive AoA-only multi-drone sensing, belief-state estimation and recurrent MAPPO. **Mismatch:** its main task is intruder interception/defense, not connectivity-constrained stationary-source localization.

**Implementation guidance for Codex:** Prioritize the mathematical assumptions, baselines and evaluation metrics in these references; record their sensing modality (DoA/AoA, RSS, TDoA), whether targets are static/moving, whether connectivity is continuous/intermittent, and which information the policy receives. Do not transfer an RF model, ground-truth observation, reward definition, or RL architecture into the project without an explicit design decision. Existing published methods mean novelty must be evaluated carefully, not assumed.

**Read tool documentation only when choosing that tool:** [Sionna RT](https://nvlabs.github.io/sionna/), [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/), [Ray RLlib multi-agent](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html), [PyTorch](https://pytorch.org/docs/stable/index.html). These links do not select an implementation stack.

## 8. Open decisions — do not treat as established requirements

The following details **have not been fixed**. Codex may explain alternatives and suggest a choice with Fausto, but must distinguish a proposal from a confirmed requirement.

**Environment and sensing**
- Map dimensions, units, coordinate and angle conventions; transmitter stationary versus moving; initial source/drone/ground-station sampling distributions.
- DoA noise magnitude/distribution/correlation, sensor orientation assumptions, missed measurements and multipath effects.
- Whether to use a more realistic RF simulator in any later stage.

**Connectivity and movement**
- Drone-to-drone and drone-to-ground link thresholds/models, whether links are symmetric and how disconnection is penalized or prohibited.
- Flight/motion dynamics, maximum displacement/speed, collisions, boundary conditions, episode length and termination.
- The exact content, timing and bandwidth of messages exchanged among drones.

**Localization, learning and evaluation**
- Final localization estimator and CRLB formulation; handling poor/degenerate geometry.
- Baseline formation families and any additional non-learning comparators.
- Agent observations, action space, reward design, centralization/decentralization, policy sharing, MARL algorithm and network architecture.
- Training budget, scenario counts, seeds, evaluation metrics and train/test splits (subject to the **required** comparison, trade-off analysis and unseen-condition testing).
- Whether/how the final project integrates optional 3D, coverage-map, wind or digital-twin components.

**Project logistics**
- Exact submission date, report format and any grading criteria beyond the stated deliverables.
- Whether there is a professor-provided code repository or reusable data beyond what is already available to the student.

## 9. Instructions for Codex

1. Treat **Section 2** as the established scope and **Section 3** as non-binding but actionable guidance. Do not downgrade MARL, connectivity or generalization into optional work.
2. You may help Fausto start implementing immediately using the recommended simplified 2D approach, without waiting for neighboring projects.
3. For every unspecified architectural or scientific choice, **label it as a proposal**, explain the trade-offs and seek Fausto's decision when consequential. Do not write an arbitrary choice into this spec as though it were official.
4. Keep hidden simulator ground truth separate from information legitimately available to agents; evaluate connectivity as **multi-hop reachability to the ground station**.
5. Keep sensing/localization, connectivity, control/learning and evaluation sufficiently modular for later changes, without assuming any particular repository structure.
6. Use the references to inform design and baselines, **not** as additional requirements. Do not claim an approach is novel merely because it combines drones, RF localization and RL; closely related published work already exists.
7. Maintain an implementation decision record **outside this source-of-truth file** as the work evolves.
