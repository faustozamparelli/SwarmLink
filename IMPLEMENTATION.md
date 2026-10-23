# SWARM-LINK — Technical System Map & Implementation Study Index

**Purpose.** Living technical implementation document and study map for Project 5: **what exists, its inputs/outputs, decisions, verified results, limitations and next study topics**. `SPEC.md` remains the source of truth for established requirements and requires Fausto's approval to change. START choices recorded here are implementation decisions agreed with Fausto, not additional supervisor-imposed requirements.

## 0. Reading conventions

- **[REQUIRED]** Established Project 5 scope or deliverable.
- **[GUIDED]** Research-team recommended starting assumption or technique; useful, **not** an imposed design.
- **[OPEN]** Unspecified: implementation needs an explicit, documented decision.
- **[RESEARCH]** Optional extension or publication-oriented investigation; not promised by the project.
- **START** = first validated end-to-end simulator (**implemented**, static experiment); **SUBMISSION** = full assessed Project 5 research implementation; **EXPAND** = credible follow-on research if time and evidence justify it.

**Do not confuse:** A requirement to *compare* or *evaluate* is fixed; numerical parameter values, libraries, learning architectures, and evaluation thresholds generally are not.

## 1. Project contract (what is actually being solved)

**[REQUIRED]** Coordinate a swarm of mobile drones to localize an **unknown radio transmitter** from **direction-of-arrival (DoA)/bearing measurements**, while keeping **every drone connected to a ground station directly or by multi-hop drone relays**. Develop a **multi-agent reinforcement learning (MARL)** coordination strategy, compare it against **fixed formations**, quantify the **localization-accuracy versus connectivity** trade-off, vary swarm size and link constraints, and test **unseen scenarios**. Deliver a trained policy/environment, comparative evaluation, simulation code, documentation and final report.

**Separation of entities:** The transmitter is the unknown **object of estimation**; the ground station is the known **communication endpoint**. They are not interchangeable. The simulator may know the source's real location; the controller does not get this as an observation.

**Not imposed by supervisors:** RF carrier frequency, antenna array, map dimensions, physical drone model, communication radius, exact DoA noise law, number of steps, number of drones, source motion, learning algorithm, model architecture, optimizer, reward coefficients, data size, and software framework. Agreed START implementation choices are recorded in §6.

### 1.1 What drones measure, consider and produce

| Stage | **Drones measure / receive** | **Drones must consider** | **Outputs** |
|---|---|---|---|
| **START** | DoA toward emitter; their own position and measurement time [GUIDED] | Geometry, noise and whether relay paths to station exist | Synthetic bearings; source estimate; connectivity graph; localization error in evaluator |
| **SUBMISSION** | Repeated DoA and information reachable through connected peers [GUIDED; exact messages OPEN] | Changing bearing geometry; graph connectivity; new estimate; limits imposed by chosen motion model [OPEN] | **Actions** (placements/movements), updated estimates, learned policy, comparative evaluation [REQUIRED] |
| **EXPAND** | More physically realistic signal/channel information, neighbor updates with delay/loss, dynamic wind inputs [RESEARCH] | Propagation, energy, collision/obstacles, imperfect communication [RESEARCH] | Higher-fidelity robustness and possible new research findings |

**Two distinct radio systems:** (A) the unknown transmitter emits a signal that drones use for **DoA sensing**; (B) drones and the ground station use **communication links** to exchange information. Receiving the transmitter does **not** imply that the drone can communicate with the station, or vice versa.

### 1.2 Three-tier scope at a glance

| Area | START — validate foundations | SUBMISSION — satisfy Project 5 | EXPAND — research possibility |
|---|---|---|---|
| World | Synthetic **2D** geometry [GUIDED] | Reproducible sampled scenarios, multiple swarm sizes/link settings [REQUIRED] | 3D, obstacles, terrain, hardware-calibrated cases [RESEARCH] |
| RF sensing | Ideal bearing + configurable noise [GUIDED] | Noisy DoA measurements with stated statistical assumptions [REQUIRED] | Array-based AoA, multipath, occlusion, heteroscedastic/correlated noise [RESEARCH] |
| Localization | Least-squares bearings [GUIDED] | Evaluated estimator and theoretical uncertainty reference; CRLB suggested [GUIDED] | Robust/Bayesian filters, realistic propagation [RESEARCH] |
| Network | Range-threshold graph [GUIDED] | Enforce/measure **ground-station multi-hop reachability** [REQUIRED] | Link budgets, packet loss, latency, capacity, relay scheduling [RESEARCH] |
| Motion | Static placements, optionally scripted movement | Sequential multi-drone decisions and measurements [GUIDED trajectory; REQUIRED learned coordination] | Realistic dynamics, energy and collision constraints [RESEARCH] |
| Baselines | Several fixed geometries [REQUIRED] | Fair comparisons vs learned policy; multiple sizes/settings [REQUIRED] | Strong constrained information-greedy / active-sensing baselines [RESEARCH] |
| Learning | No RL needed to validate measurement pipeline | MARL formulation, training, learned control, evaluation [REQUIRED] | Constrained MARL, graph policies, communication learning, transfer [RESEARCH] |
| Generalization | Independent test random seeds | Unseen transmitter locations, sizes **or** link conditions [REQUIRED] | Cross-topology, cross-scene, sim-to-real robustness [RESEARCH] |
| Integration | Entirely standalone Python [GUIDED] | Module boundaries so other work can connect; no full digital twin imposed | Sionna RF maps, AIRWISE flight-cost field, external simulator [RESEARCH] |

---

## 2. System objects and data flow

### 2.1 Core entities and their information

| Entity / record | Holds (conceptually) | Must **not** be confused with |
|---|---|---|
| **Source truth** `s_true` | Actual transmitter `(x,y)` in simulation; scenario ground truth | Policy observation / estimated source position |
| **Drone i** `p_i,t` | Position, time index, optional velocity/heading and status | Its inferred knowledge of all other agents |
| **Ground station** `g` | Fixed reference endpoint position in the starter world [position/time behavior OPEN] | Unknown transmitter |
| **DoA measurement** `z_i,t` | Bearing angle, time, drone ID, optional validity/variance | Range or direct transmitter coordinates |
| **Communication graph** `G_t` | Vertices `{ground, drones}`, edges for available links | Sensing geometry / actual RF signal direction |
| **Measurement store** `M_t` | Current and/or historical bearings with known measurement positions | The true hidden state |
| **Estimator output** `b_t` | Estimated source, uncertainty/quality flag, possible belief distribution | Guaranteed correct source location |
| **Agent observation** `o_i,t` | Legitimately accessible local state and received neighbor messages | All privileged simulator state |
| **Agent action** `a_i,t` | Proposed movement, waypoint or other control command [OPEN] | Guaranteed feasible physical transition |
| **Policy / controller** `π` | Maps observations to actions; shared or per-drone [OPEN] | Localization estimator or simulator |
| **Episode result** | Outcomes, network history, errors, actions, seed and configuration | Training reward alone |

### 2.2 End-to-end dependency chain

```text
Scenario generator
  -> hidden source truth, drone start positions, ground station, world/noise/link parameters
  -> DoA simulator: (source truth, drone position, noise) -> angle measurements
  -> Connectivity: (drone positions, ground station, link model) -> graph, reachability
  -> Information exchange: (graph, per-drone measurements/known positions) -> allowed messages
  -> Localization estimator: (available measurements, sensor positions) -> estimate + quality
  -> Observation builder: (own state, allowed messages, local/aggregate estimate) -> o_i,t
  -> Coordination policy / baseline: observations -> simultaneous drone actions
  -> Motion/feasibility model: (positions, actions) -> new positions
  -> [repeat measurement, graph, estimation and policy at time t+1]
  -> Evaluator (may access hidden truth): localization, connectivity, movement, learning results
```

**Feedback loop:** Drone placement influences both **bearing geometry** (information quality) and **graph edges** (connectivity). Measurements update the estimate, which changes the desirability of the next placement. This coupling is the project, not three unrelated demos.

**Implemented START order:** generate one static scene → measure each drone's bearing → build the graph → forward only station-reachable, finite measurements → ground-station LS estimate → evaluate truth-based error and strict connectivity. One snapshot; no movement, history, controller or policy observations. The complete feedback loop above remains future work; communication/movement timing and distributed estimation remain [OPEN] before training.

**Permanent rules vs benchmark choices:** sensing, graph reachability, LS and scoring accept arbitrary valid positions; the evaluator takes supplied drone/source positions and a configurable station. Fixed formations, a common centre, safe spawning envelope and source priors exist only in scenario generation/configuration. They do not constrain future motion, require formation preservation, keep drones near their spawn, or require sources inside a formation. Disconnected states remain representable and fail connectivity. Q is an evaluation metric, not a prescribed RL reward; no RL interfaces were added.

**Implemented sequential order (`sequential-v1`):** measure/deliver/estimate at t=0 → for each of T steps: open-loop displacement → map-boundary truncation → new bearings/graph → append currently deliverable data → cumulative-history LS → log. Episodes continue through T despite outages. Controllers receive only team size, motion budget and independent RNG; truth remains restricted to sensing/evaluation. No formation must be preserved.

---

## 3. Component cards (inputs → operations → outputs → metrics)

Each card preserves the same **START / SUBMISSION / EXPAND** distinction. The algorithm lists are **candidates**, not selections.

### 3.1 World / scenario generator

- **Inputs:** configuration and random seed; region bounds; `N` drones; station location; initial formations/poses; source location; noise and link parameters.
- **Outputs:** complete simulator truth, initial positions, reproducible scenario identifier/configuration.
- **Fixed:** vary drone counts and connectivity conditions and evaluate unseen cases [REQUIRED]. Synthetic data creation and 2D start [GUIDED].
- **START implemented:** independent random streams for transmitter, deployment centre, orientation, angular noise and random placement; shared scenarios across all formations, sizes and ranges. **`randomized-v1`** retains broad independent source/deployment sampling; **`controlled-v1`** fixes the deployment centre and samples the source uniformly in area within a configurable nearby disk. No formation-specific hull conditioning. Every trial is retained, including disconnected/invalid cases. The common placement envelope prevents boundary violations without clipping or outcome-dependent rejection. Defaults: §6.
- **SUBMISSION:** independent train/validation/test scenario generation; explicit sampling distributions and coverage of difficult cases; record configurations for reproducibility.
- **EXPAND:** 3D terrain, buildings, occlusions, wind, heterogeneous sensors, adversarial or out-of-distribution scenarios.
- **Metrics/checks:** percentage valid/generated; scenario diversity; dataset split independence; no hidden source coordinate passed to policy.
- **[OPEN] beyond START:** moving sources, different priors/boundaries and sampling distributions; no claim of environmental generalization from current trials.

### 3.2 Drone geometry and motion

- **Inputs:** positions `p_i,t`, commanded actions `a_i,t`, time step, boundaries and constraints.
- **Outputs:** new positions `p_i,t+1`, displacement/travel cost, feasibility flags.
- **Fixed:** drone formations and learned placement/coordination must be studied [REQUIRED]. No flight dynamics fixed.
- **START implemented:** fixed placements only; no motion, action space or formation optimization. All formation centroids and RMS radii match; maximum extent may differ because geometry differs.
- **Sequential implemented:** point-agent displacements with norm ≤ `vmax × Δt`; rectangular-wall clipping reduces actual movement and is logged. Travel sums actual displacements. No connectivity rejection, early outage termination, formation constraint, collision avoidance or acceleration model. Provisional defaults: §6; no validated UAV/training specifications are implied.
- **SUBMISSION:** executable sequential movements, measurements after motion and multi-agent actions, with action feasibility recorded.
- **EXPAND:** velocity/acceleration/yaw limits, obstacle avoidance, collision prevention, energy/wind flight-cost maps.
- **Candidates:** discrete grid steps; bounded continuous displacement; waypoint commands; kinematic step; constrained projected action.
- **Metrics:** path length [distance], final placement, steps to accuracy threshold, invalid/blocked action rate, optional energy.
- **[OPEN] for future learned control:** reward, constraint handling and richer dynamics. Current sequential moves may disconnect; only speed and map bounds constrain movement.

### 3.3 DoA sensor and measurement noise

- **Inputs:** hidden transmitter `s`, drone pose `p_i,t`, heading/reference frame, `σ` or chosen sensor model, seed.
- **Outputs:** measured bearing(s) `z_i,t`, angle convention, uncertainty/validity metadata.
- **Fixed:** DoA/bearing sensing [REQUIRED]; no range or RSS measurement is imposed. Gaussian angle noise [GUIDED].
- **START implemented:** global-frame `atan2` bearing plus independent wrapped Gaussian noise; radians internally, known positions/headings. Identical per-drone angular noise is reused across matched formations/ranges. Source–sensor separation ≤ `1e-9 m` gives an undefined measurement, excluded with a logged missing-measurement count.
- **Sequential implemented:** one bearing/drone at t=0 and after each move; independent noise across times, paired by scenario/step/drone across controllers. Equal generated sensing budgets do not imply equal delivered histories. Repeated-noise independence is an uncalibrated assumption.
- **SUBMISSION:** characterize impact of noise over source locations and formations; log observations/variance; unit and frame consistency.
- **EXPAND:** realistic antenna arrays, multipath/non-line-of-sight, bias/outliers, distance-dependent error, dropouts, joint noise.
- **Candidates:** direct `atan2` bearing; wrapped-Gaussian angular noise; von Mises noise (extension).
- **Metrics:** angular residual [radians/degrees], failure/dropout rate, estimator sensitivity to noise.
- **[OPEN] beyond current implementation:** heading errors, correlated measurements, covariance models and transmitter detectability. Sensing does not depend on station connectivity; delivery does.

For a 2D point `s=(x_s,y_s)`, `p_i=(x_i,y_i)`, ideal bearing:

`h_i(s)=atan2(y_s-y_i, x_s-x_i)`; illustrative noisy observation `z_i = wrap(h_i(s)+epsilon_i)`.

**Numerical safeguard:** angular residuals must be wrapped; naive subtraction makes angles near `-π/+π` look far apart. A bearing direction alone does not determine range.

### 3.4 Communication links, graph and exchange

- **Inputs:** drone positions, ground station location, chosen radio/link model, network/message history.
- **Outputs:** adjacency, connected components, ground-station reachability per drone, allowed message routes.
- **Fixed:** *every* drone must reach the ground station directly or through other drones [REQUIRED].
- **START implemented:** undirected `distance <= R` edges, including the station; BFS determines each drone's shortest station hop count. Also compute **drone-only** components, excluding paths through the station. Classes: `direct` (one station hop), `relay` (>1), `peer_only` (unreachable with drone neighbors), `isolated` (unreachable without neighbors). A directly connected drone with no drone neighbors is still `direct`.
- **START reporting:** per-drone class/component/reachability/hops and estimator use; number of unreachable drones; full-swarm feasibility. **Any unreachable drone fails connectivity**, independently of localization. Unreachable measurements never enter the station estimator.
- **Sequential reporting:** terminal connectivity requires all drones reachable at T; continuous connectivity requires this at every evaluated snapshot 0…T. Temporary outages/reconnection are allowed. Per-drone and swarm traces retain connected fractions, transition events, outage exposure and longest outage (§4). Terminal success never proves continuously maintained connectivity or reachability between snapshots.
- **SUBMISSION:** evaluate/maintain reachability at appropriate time steps; restrict distributed information to connected pathways; vary link conditions.
- **EXPAND (documentation only):** per-hop latency, packet loss, link quality, asymmetric/RF-dependent connectivity, obstacles, bandwidth/interference, routing and learned messages. None is implemented.
- **Candidates:** breadth-first search / union-find for components; graph traversal from ground station; later graph policies.
- **Metrics:** `C_t = 1` iff **all** drones can reach ground; fraction of connected steps/episodes; outage counts/duration; relay hops; feasible region.
- **[OPEN] beyond START:** heterogeneous/station-specific ranges, continuous movement connectivity and communication delay. START uses one shared radius, one static instant and instantaneous lossless forwarding within the station component.

**Important:** Fully interconnected drones disconnected from the station fail that snapshot's station reachability. Sequential reconnection may change the terminal outcome; report continuous connectivity separately. A neighbor does not imply a station route.

### 3.5 Transmitter localization estimator / belief

- **Inputs:** bearings, sensor positions at measurement times, measurement uncertainties, accessible measurement history.
- **Outputs:** `s_hat` estimate, optional covariance/uncertainty, validity/degeneracy flag.
- **Fixed:** localization error must be evaluated across geometries and source positions [REQUIRED]. Least squares [GUIDED].
- **START implemented:** bearing-line LS via `numpy.linalg.lstsq`, centred on sensor positions. Only station-reachable, finite measurements are used. Detect insufficient data, rank deficiency (`rcond=1e-12`), numerical ill-conditioning (`condition > 1e6`), nonfinite inputs/solutions and solver failure.
- **Ray check:** require positive projection onto every measured ray (`>1e-9 m`); otherwise `direction_inconsistent`. Log backward-ray count and wrapped angular residual RMS. Invalid estimates have **no coordinates/error**, rather than a fallback location. Valid estimates outside the map are not clipped. Nonlinear angular LS remains future work.
- **Sequential history:** cumulative centralized `(acquisition position, bearing)` data, with timestamp/drone identity in logs. Direct/multihop deliveries arrive immediately; previously delivered data survives later outages. Offline measurements are unavailable and never uploaded retrospectively. Reconnection contributes new measurements only. Existing LS/validity checks are unchanged; history can support a valid estimate during a current outage.
- **SUBMISSION:** estimator with noise-aware evaluation; handle degenerate geometry; optionally nonlinear angle least squares and uncertainty estimate; compare same estimator for all strategies.
- **EXPAND:** robust M-estimators, Bayesian filters, particle filters, dynamic transmitter tracking, distributed consensus/belief fusion.
- **Candidates:** weighted line-distance LS; nonlinear wrapped-angle LS; maximum likelihood; EKF/UKF/particle filter for sequential beliefs.
- **Metrics:** Euclidean source-position error `e_t = ||s_hat_t - s_true||` [m]; RMSE/median/percentiles; failure and calibration rates; inference cost.
- **[OPEN] beyond current implementation:** sliding-window history, local buffering/delayed upload, distributed estimation, priors and calibrated uncertainty. Static START still uses one snapshot; sequential history assumes a stationary source. Numerical/ray tolerances remain engineering defaults.

**Pitfall:** Closely parallel bearings can make the estimate unstable. The system needs an explicit “unobservable/ill-conditioned” outcome, not an apparently precise coordinate.

### 3.6 Geometry, Fisher information and CRLB

- **Inputs:** positions of measurement sensors, assumed source location for evaluation, bearing Jacobian and noise covariance.
- **Outputs:** Fisher information matrix `J`, CRLB if defined, geometry-quality indicator.
- **Fixed:** fixed-formation comparison [REQUIRED]. CRLB is specifically recommended as a benchmark [GUIDED], not an officially selected reward.
- **START implemented:** scene plots and LS rank/conditioning diagnostics. Fisher information, CRLB computation and Jacobian verification are **not implemented**; retain these as study topics before selecting future information-based criteria.
- **SUBMISSION:** benchmark or diagnostic information bound for valid noise assumptions; analyze accuracy versus geometry/connectivity, report singular cases.
- **EXPAND:** uncertainty-averaged/robust information design, Bayesian CRLB, multi-step information planning, information surrogates.
- **Candidates:** `trace(J^-1)` (A-optimal), `-log(det J)` / maximize `log det J` (D-optimal), worst-axis eigenvalue objectives; choose and justify.
- **Metrics:** position-error bound [m²] or root bound [m], determinant/condition number with correct units/convention, estimator gap to bound.
- **[OPEN]:** evaluation point for `J` when source is unknown; noise covariance model; geometric design criterion.

For independent bearings with angle variance `σ_i²` and `d_i²=(x_s-x_i)²+(y_s-y_i)²`:

`∇_s h_i = [-(y_s-y_i), (x_s-x_i)] / d_i²`,  
`J = Σ_i (1/σ_i²) (∇h_i)(∇h_i)^T`,  
`Cov(s_hat) >= J^-1` **only under applicable regularity/unbiasedness conditions and when `J` is invertible**.

**Crucial information boundary:** The simulator may compute an *oracle* CRLB at the actual source for evaluation. A deployed policy cannot obtain that oracle criterion unless it knows the source; planning must use legitimate estimates/beliefs or another non-oracle proxy.

### 3.7 Static and classical movement baselines

- **Inputs:** shared scenario distribution, formation definition or scripted/optimized movement rule, sensing/estimator/network model.
- **Outputs:** baseline trajectories/positions and per-scenario experiment results.
- **Fixed:** compare learned coordination against **fixed formations** [REQUIRED]. Additional non-learning movement baseline [RESEARCH but methodologically valuable].
- **START implemented:** `line` (equally spaced), `polygon` (regular), `compact` (nearest square-lattice sites) and `random` (disk samples, recentered/rescaled). Formation generation never receives the transmitter. Random orientation and placement streams are independent of source truth. Same scenario ID, station, centre, bounds, RMS radius, N, range and per-drone noise across methods; equal one-bearing-per-drone budget.
- **Sequential implemented:** `hold_still` (zero movement) versus `random_movement` (independent uniform-area displacement-disk draws per drone/step). Same initial source/station/positions, noise, horizon, sensing budget and maximum movement allowance; actual travel is reported separately. Initial formation is configurable and never an ongoing constraint. Neither controller receives truth, beliefs or graph/global state; no deliberate active-sensing/reconnection strategy is claimed.
- **SUBMISSION:** same initial conditions/test instances, sensors, movement budget, estimator and connectivity tests as learned policy; comparisons across swarm sizes and link constraints.
- **EXPAND:** connectivity-aware greedy Fisher-information planning, receding horizon / constrained optimization, classical active-sensing strategies from published work.
- **Candidates:** fixed compact/line/polygon placements; random feasible control; information-greedy next waypoint; constrained optimizer.
- **Metrics:** error, connectivity, feasible-rate, travel, steps, statistical variation; advantage over each baseline.
- **[OPEN] beyond START:** alternative formation scales/families and fair resource matching against sequential policies. Current disconnected or invalid placements are retained and scored zero; no optimization or feasibility resampling.

**Incremental benchmarking principle (future context):** progress may compare random placement → fixed geometry → classical static optimization → classical sequential active sensing → RL → more advanced learning/realism, when evidence or literature justifies each step; none is an automatic implementation commitment. Deployment methods cannot access source truth; future controllers may use their own observations/estimates and legitimate messages. Match sensing/movement budgets and evaluation conditions. For each evaluated change, record method/information, comparator, rationale, configuration/protocol version, accuracy/connectivity/failures/Q, improvements/regressions/uncertainty and limitations. Preserve previous results; incompatible protocol scores require separate interpretation. Future RL may move each drone independently.

**Fairness:** Static and moving methods have different measurement/travel budgets. State whether results compare equal *time, number of bearings, path length, steps*, or another controlled resource.

### 3.8 MARL problem formulation

- **Inputs:** simulator transitions, local observations/messages, joint actions, reward and termination rules.
- **Outputs:** trainable multi-agent environment, observation/action/reward traces, learned policies.
- **Fixed:** multi-agent RL, observation/state/action/reward specification and learned coordination [REQUIRED]. No specific RL algorithm mandated.
- **Current status:** static evaluation and the sequential motion/history loop with two open-loop baselines are implemented/tested. No MARL library, training, RL adapter or agent-observation interface exists. Learned observations/reward/constraint handling and deliberate active sensing remain future decisions.
- **SUBMISSION:** train a policy and compare to fixed formations under matched assumptions; evaluate on held-out scenarios; document training curves and seed variation.
- **EXPAND:** constrained RL, decentralized execution with centralized training (CTDE), graph/invariant architectures, variable-sized teams, learned communications, curriculum/transfer.
- **Algorithm candidates:** independent PPO, parameter-sharing PPO, MAPPO/CTDE, QMIX for suitable discrete cooperative actions, centralized joint-action RL as a diagnostic **(not automatically decentralized MARL)**; constrained MARL methods when hard connectivity is central.
- **Metrics:** learning curve, rollout localization error, network feasibility, policy inference cost, robustness/generalization, variance across seeds.
- **[OPEN]:** actions, local/global observations, sharing/communication protocol, reward equation/coefficients, hard vs soft connectivity constraint, algorithm, centralized critic, termination rules, model input for variable `N`.

**Separate three information levels:** (1) hidden world truth, (2) information legitimately visible to each executing agent, (3) extra global information optionally available to a *training critic*. A centralized training critic does not justify revealing source truth to the policy at evaluation.

**Reward/evaluation are not equivalent.** A possible reward could use estimator uncertainty, improvement, movement and connectivity; do not declare one fixed. The final published metrics must be measured separately from whichever training reward is selected. Simulator truth can legitimately generate training rewards, but must never leak as a policy observation.

**Connectivity constraint:** A finite penalty does **not** mathematically guarantee connectivity. If “remain connected” is interpreted as a hard invariant, enforce/verify feasibility in transitions or use a constraint-handling method; separately log infeasible attempts.

### 3.9 Evaluation, testing and reproducibility

- **Inputs:** frozen scenarios/test seeds, trained checkpoints, baseline policies, experiment metadata.
- **Outputs:** comparison tables/figures, failures, confidence/variance, reproducible artifacts.
- **Fixed:** comparison by swarm size and communication requirements; accuracy-vs-connectivity trade-off; at least one unseen condition class [REQUIRED].
- **START implemented:** standard-library `unittest`, paired experiment runner, per-trial/per-drone CSVs, setting/formation summaries, paired comparisons, metadata and scene/metric PNGs. Controlled and randomized runs use separate directories/leaderboards. Reports label the benchmark, display source/estimate/error and individual connectivity, and include the first disconnected case (diagnostic selection only; no statistical exclusions). Reference-distance sensitivity rescores the same trials at `0.01/0.1/1/10 × diagonal`, with scenario-bootstrap mean-Q intervals; primary Q is unchanged. See README for commands/artifact links.
- **SUBMISSION:** structured test matrix, multiple randomized scenarios, identical held-out instances for each method, sufficient seeds, explicit statistical reporting and ablation of chosen assumptions.
- **EXPAND:** robust OOD benchmark, hardware-in-the-loop, realistic channel data, independent reproduction, statistical sample-efficiency studies.
- **Metrics:** see §4; log failure codes, seeds, versions, model checkpoints, configuration.
- **[OPEN] beyond START:** training/validation/test budgets and splits, future stopping criteria, confirmatory statistical analysis, unseen dimensions and training compute limits.

### START verification and findings — randomized-v1 baseline, 2026-10-23

- **Original checks:** **28 tests passed**: noiseless recovery, wrapping, rays, rank/conditioning failures, relays, peer-only components/isolation, information filtering, deterministic seeds, matched scenarios/scales, scoring/denominators, inconclusive equal-score comparisons and generated artifacts. Two full runs produced identical SHA-256 hashes for `trials.csv` and `drones.csv`. Saved-output audit: **36,000 trials**, **9,000 complete four-formation cases**, **192,000 drone observations**. Original artifacts remain in `results/`; new reports are kept separately.
- **Actual results:** compact/polygon/random/line mean Q = **0.5330 / 0.5252 / 0.5127 / 0.4852**; full-swarm connectivity = **54.77% / 53.97% / 52.91% / 52.29%**. Full tables, conditional accuracy, CIs and failure counts: `results/summary.md`, README and `results/summary.json`.
- **Interpretation:** compact has the highest empirical mean here; its paired advantage over polygon is **0.00782**, exploratory 95% CI **[0.00400, 0.01154]**. This is a small distribution-specific difference, not universal optimality. Communication range dominates feasibility: compact reaches **19.90% / 55.63% / 88.77%** full connectivity at **200 / 350 / 500 m**. A disconnected trial remains failed even with a valid accurate partial-network estimate.
- **Failures:** most all-trial estimator failures are insufficient reachable measurements. Among fully connected trials, estimator failure rates are **0.020% / 0.041% / 0.231% / 3.506%**, respectively; direction-inconsistent fits account for these. Line's conditional RMSE is **187.92 m**, versus compact's **65.41 m**: its heavy tail is not visible from median error alone.
- **Limits:** one empty square map and station; point sensors, known global heading, independent fixed-variance noise and ideal instantaneous links. No physical RF/multipath model, flight dynamics, trained policy or environmental-generalization evidence. Conditional errors use different surviving subsets across formations and cannot replace all-trial Q. Noise can mask underlying near-parallel geometry; positive ray projection is a diagnostic, not an angular-likelihood fit. Bootstrap intervals are exploratory, conditional on the selected settings, without multiplicity correction.
- **Future improvements (documentation only):** study CRLB/estimator bias and angular LS; then specify sequential movement/active sensing and MARL. Wind-aware movement costs, terrain/obstacles, richer propagation and held-out environment layouts/map families remain later stages. Distinguish unseen transmitter/drone scenarios **within a map** from genuinely unseen environment families; no extension infrastructure was added.
- **Study next:** singular values versus geometric observability; line-distance versus angular residuals; station BFS versus drone-only components; common random numbers, paired bootstrap and selection bias in conditional metrics. Reading pointers: §8 [L1], [L2], [L6]; these are study targets, not claims of reproduced methods.

### START controlled/randomized refinement — 2026-10-23

- **Change/comparator:** same four placement methods and information restrictions; only the deployment/source prior changes. `controlled-v1` diagnoses near-source geometry at a fixed site, whereas `randomized-v1` measures robustness across broad source/deployment draws. Same settings/noise/formation scales; no combined leaderboard. Metadata records sampling region, protocol/config hash, methods and `static-report-v2`. Existing randomized raw data is exactly preserved. This checkout had no prior L-sensitivity/disconnected-scene implementation; both were added as small reporting extensions, without new modules/dependencies.
- **Actual controlled results:** polygon/compact/random/line mean Q **0.9766 / 0.9718 / 0.9213 / 0.8482**; full connectivity **97.90% / 97.66% / 92.68% / 86.96%**; connected-valid mean error **3.45 / 3.48 / 3.80 / 3.87 m**, RMSE **4.13 / 4.20 / 5.15 / 6.12 m**. Connected-only estimator failure **0 / 0.239 / 0.324 / 2.185%**. Full constituent metrics, seed variation and CIs are in each generated summary.
- **Interpretation:** polygon−compact paired Q difference **0.00479**, exploratory 95% CI **[0.00129, 0.00846]**; randomized still favors compact empirically. The changed leader shows dependence on evaluation conditions, not a universal winner or a newly improved method. Nearby-source/fixed-site cases are easier; absolute scores across benchmarks do not measure method improvement. Both mean-Q orders are stable over tested L scales; randomized compact/polygon means are very close at `0.01 L`, without a significance claim there. With diagonal L much larger than controlled errors, primary Q is dominated by connectivity/validity.
- **Verification:** 36 tests pass (all original 28 plus eight focused sampling/fairness/core-reuse/output tests). Each full run has 36,000 retained trials, 9,000 complete matched cases and 192,000 drone observations; every saved Q and each controlled source/centre checked. Two controlled runs reproduce all nine CSV/JSON artifacts exactly. Randomized trial/drone logs match the original bytes; legacy artifacts and SPEC.md remain unchanged. Scene, disconnected-case, metric and sensitivity plots inspected. Plot review caught an overflowing failure label; splitting titles fixed it without changing numerical results. No failing checks remain.
- **Limitations:** one controlled site and a selected source radius, uniformly random shared orientations, equal RMS but differing extents; no site/radius robustness claim. No formation-conditioned sampling or failed-case rejection. Conditional localization metrics retain different success subsets. Sensitivity is diagnostic rescoring, not tuning the primary metric. No failing algorithmic approach was introduced; no new estimator, optimizer, movement or MARL code exists.
- **Postponed investigation — documentation only:** analyze extreme line-formation errors; inspect bearing geometry, rank, singular values and condition numbers; distinguish geometric degeneracy, noise sensitivity and estimator limitations. Consider alternative estimators or geometry-aware placement optimization only if justified. Relevant later for stronger formations, classical active sensing and fair RL comparisons; no such investigation or optimization was performed in this iteration. Study pointers: §8 [L1], [L2], [L6], and paired experimental design/source-prior effects.

---

### Sequential movement and cumulative history — sequential-v1, 2026-10-23

- **Implemented change:** reused existing initial samplers, sensing, graph, bearing-line LS and Q helper; extended existing experiment/connectivity/report/CLI modules only. All generated episodes/steps are retained. Static scoring and saved N=3/5/8 artifacts are unchanged. Full numerical evidence is in the separate sequential reports linked from README.
- **Main configuration:** 200 paired controlled scenarios (8 seeds × 25), polygon starts, N=5/T=60, ranges 200/350 m: **800 episodes**, **48,800 snapshots**, **244,000 drone snapshots**, 120 simulated seconds and 305 generated bearing slots per episode. This diagnoses blind movement from informative initial geometry; it does not establish training requirements.

| Controller | Mean Q | Terminal % | Continuous % | Final mean / median / RMSE [m] | Mean total travel [m] |
|---|---:|---:|---:|---:|---:|
| hold_still | 0.999694 | 100.00 | 100.00 | 0.434 / 0.385 / 0.499 | 0 |
| random_movement | 0.907209 | 90.75 | 84.00 | 0.454 / 0.422 / 0.514 | 1999.23 |

- **Interpretation:** 27/400 random episodes finish connected after an earlier outage and receive positive final Q; 37 finish disconnected and score zero despite valid historical estimates. Random movement records **598 per-drone disconnection / 460 reconnection events**. Main final estimates are all valid. Paired Q(random−hold) is **−0.09248**, exploratory 95% CI **[−0.11998, −0.06749]**. Blind motion does not help this near-source, initially informative geometry; this says nothing conclusive about deliberate active sensing or RL. At 200 m, random terminal/continuous rates are **81.5% / 68%**, versus 100%/100% at 350 m. Conditional error subsets differ; Q includes every episode.
- **Debug run:** N=3/T=30, 2 seeds × 5 scenes, same ranges: 40 episodes over 60 simulated seconds. Hold terminal/continuous **90%/90%**; random **85%/65%**. Its paired-Q interval crosses zero; this small run verifies behavior rather than establishing performance.
- **Verified:** **50 tests pass** (36 earlier + 14 sequential): t=0/budgets, acquisition poses/history retention, no retrospective upload, terminal-vs-continuous scoring, persistent outages without early exit, speed/wall/travel limits, pairing/repeatability, source-blind actions, variable N/horizons, denominators and static-output protection. Hashes preserve prior static artifacts, SPEC.md, locked dependencies and sensing/LS code. RNG purpose/time keys are separate (the initial offset scheme could collide at long horizons); the helper was minimally extended while preserving legacy draws. Empty-estimate/CDF plots were clarified; no new algorithm was introduced. Final audit: all eight main CSV/JSON outputs reproduce exactly; all 44 protected files other than the deliberately extended RNG helper remain unchanged. Saved main/debug logs verify every history count, paired noise/initial position, speed/travel limit, outage/event definition, terminal/continuous outcome and Q. No failing checks remain.
- **Limits/future alternatives:** ideal point motion, known positions/headings and a static source; no collision/acceleration, wind, terrain or energy model. Independent repeated noise helps hold-still averaging and remains uncalibrated. Cumulative history grows with T; sliding windows, local buffering/delayed upload, distributed estimation, per-hop latency, packet loss, link quality, obstacles and RF-dependent links remain documentation only. Connectivity/exposure are sampled approximations, not continuous-flight guarantees. No large-N/horizon study or new estimator was introduced; the extreme-line-error investigation above remains postponed.
- **Future controller information:** deliberate reconnection may need communication status, neighbors, past connected positions or routing. Reversing the last action cannot guarantee a route in a changing graph. Exact RL-agent observations, reward and constraint handling remain open; no privileged-state interface was added. Study: temporal bearing observability, cumulative-noise assumptions, graph transitions, outage sampling and paired trajectories; §8 [L1], [L2], [L6] are reading targets, not replicated algorithms.


## 4. Metrics dictionary — **define denominators before coding**

| Metric | Definition or reporting convention | Use / caveat |
|---|---|---|
| **Instant position error** | `e_t = ||s_hat_t - s_true,t||_2` [m] | Requires estimator success; define what happens on failure. |
| **START mean error / median / RMSE** | Euclidean error; mean, median and `sqrt(mean(e²))` over **valid, fully connected** trials | Always publish `accuracy_trials`, disconnected exclusions and connected-invalid exclusions. No valid cases → missing metric, never zero error. |
| **RMSE** | `sqrt(mean(e²))` over explicitly identified cases/times [m] | Never silently mix per-episode terminal RMSE and all-timestep RMSE. |
| **Median / p90 error** | Quantiles of `e` [m] | Expose heavy-tailed errors/degenerate geometry. |
| **Localization success** | No thresholded localization success in START | A future sequential stopping threshold would require a separate decision; never use it to define connectivity. |
| **START overall empirical quality** | `Q=L/(L+error)` when valid **and** fully connected; otherwise `Q=0`; overall = arithmetic mean over **all** trials | `L` fixed before the run, shared by all formations, initially map diagonal. Continuous accuracy reward, no hard error cutoff; empirical comparison metric, not a theoretical objective or selected RL reward. |
| **L sensitivity** | Rescore identical errors/connectivity/validity at `0.01/0.1/1/10 × diagonal`, keeping every trial and paired scenario blocks | Separate diagnostic, with mean-Q CIs; does not modify primary Q or merge benchmarks. Close mean orders do not establish significance. |
| **Estimator failure rate** | Invalid station estimates / **all trials**; additionally invalid / **fully connected trials** | Publish failure reasons, including insufficient reachable measurements. These denominators answer different questions. |
| **Per-drone connectivity** | Direct/relay/peer-only/isolated counts, each drone's station reachability and drone-only component | Drone-observation denominator; also save per-ID/per-setting percentages. |
| **Unreachable drones** | Count per trial, total and mean per formation | Any positive count fails full-swarm connectivity, regardless of error. |
| **CRLB / SPEB** | `trace(J^-1)` [m²], or `sqrt(trace(J^-1))` [m] | Under specified DoA noise model; oracle vs estimated-source evaluation must be labeled. |
| **All-connected indicator** | `C_t=1` if every drone has a graph path to the station | Core constraint; not average node degree. |
| **Connectivity rate** | Static: `100 * fully_connected_trials / all_trials`; sequential: separate terminal %, continuous % and connected-snapshot fractions | Connectivity never depends on localization-error thresholds. |
| **Episode connectivity** | Fraction of episodes with `C_t=1` at **all evaluated steps** | Useful for hard constraint compliance. |
| **Sequential terminal / continuous** | Terminal: all drones station-reachable at T. Continuous: all reachable at every evaluated snapshot `0…T` | Temporary outages are allowed; never equate terminal success with maintained connectivity. No early outage termination/rejection. |
| **Sequential connected-step fraction** | Per-drone mean reachability over **T+1 snapshots**, including t=0; swarm fraction uses all-drone reachability | Keep individual reachability separate from full-swarm outcomes. |
| **Sequential outage exposure / events** | Total/longest disconnected left-endpoint intervals `0…T−1` × Δt; events use adjacent reachability transitions | Exposure ≤ TΔt. Initial outage is not a disconnection event; final-only outage adds no observed interval. A stated discrete approximation. |
| **Sequential final-error Q / accuracy** | `Q=L/(L+e_T)` iff terminal-connected and final estimate valid; otherwise 0; mean across **all episodes**. Mean/median/RMSE use final valid, terminal-connected episodes | No error cutoff or continuous-connectivity gate. Preserve exclusion counts/failures. Separate experimental convention; RL reward remains open. |
| **Connectivity outage** | Number/duration of `C_t=0` intervals | Define discrete sampling / continuous move semantics. |
| **Travel cost** | `Σ_t Σ_i ||p_i,t+1-p_i,t||` [m] | Not equivalent to energy if wind/flight dynamics omitted. |
| **Steps to target** | First `t` meeting the localization threshold | Treat failures/censored episodes consistently. |
| **Generalization gap** | Held-out minus training/validation performance for same metric | Do not use test data for tuning. |
| **Runtime** | Simulator/sensing, estimation, training and policy inference separately | Hardware/device specification if compared. |
| **Uncertainty quality** | Predicted vs observed errors, covariance coverage where available | Only if estimator outputs meaningful uncertainty. |

**Mandatory experiment axes:** multiple swarm sizes; formation geometry; communication constraints; learned vs fixed coordination; unseen transmitter locations, swarm sizes **or** link conditions. Sensor noise studies are scientifically important and supported by initial guidance, but no fixed numeric levels were assigned.

**Minimum figure categories:** geometric scenes/trajectories with links, estimation error distributions, accuracy-vs-connectivity curve, performance across swarm sizes/link settings, MARL training curve and held-out evaluation. These figure types are **proposed reporting practice**, not supervisor-specified chart formats.

## 5. Candidate software boundaries (interfaces, **not** a mandated API)

Keeping these records separate prevents simulator truth from contaminating learned observations.

```text
ScenarioConfig {seed, N, world, sensor_model, link_model, motion_model, horizon}
TruthState     {source_true, drone_poses, ground_station_pose, time}
Bearings       {time, drone_id, drone_pose_at_measurement, angle_rad, valid, uncertainty}
NetworkState   {time, adjacency_with_ground_station, each_drone_reachable, all_reachable}
Estimate       {source_hat, uncertainty?, valid, diagnostics}
AgentObs[i]    {self_pose, own_measurements, allowed_neighbor_data, accessible_estimate?}
JointAction    {action_for_each_drone}
StepOutput     {new_observations, reward(s), done_flags, diagnostics}
EvalRecord     {seed, scenario_id, method, time, error, graph_status, travel, failures, ...}
```

**Future field names and optional members above remain illustrative [OPEN].** The *separation of responsibilities* is the important design property.

**Actual code:** same module structure: `scenarios.py` (initial scenarios/formations), `localization.py` (unchanged sensing/LS), `connectivity.py` (graph plus reachability-time metrics), `experiment.py` (static trials and sequential motion/history/baselines/summaries), `report.py` (exports/plots) and `__main__.py` (CLI). No new modules/dependencies, controller hierarchy, RL adapter or map abstraction. Controllers receive only `controller_displacements(controller, n_drones, rng, max_displacement)`; other future records above remain conceptual. Ground truth stays in scenario generation, sensing and evaluator records/plots.

Implemented invariants/tests (future RL observations remain undecided):

1. Zero-noise, non-degenerate bearings recover the source within numerical tolerance.
2. Angular wrapping gives small residual across the `±π` boundary.
3. A network of drones connected only to each other but not the station is **disconnected for the mission**.
4. A multi-hop chain reaching the station is **connected**, even when the farthest drone lacks a direct station link.
5. Collinear/near-parallel rays are flagged as ill-conditioned; no false certainty.
6. The hidden source coordinate never enters an execution policy observation.
7. Sensor positions are associated with the correct timestamps in sequential fusion.
8. Identical seeds/configurations reproduce a rollout (subject to documented library determinism).
9. Evaluation can compare methods on the exact same sampled scenarios.
10. Each reported metric has a denominator and explicit handling of missing estimates/disconnected episodes.

## 6. Open scientific decisions to review with Codex (do **not** silently settle)

The START choices below are **agreed implementation decisions**, not supervisor requirements. Numerical defaults are configurable in `Config` (common run dimensions also exposed by the CLI); metadata records the exact configuration and its hash.

| Decision | Why it matters | Status |
|---|---|---|
| START world/source | Distinguish local geometry from broad deployment robustness | **Implemented:** same `[0,1000]² m` map and station `(500,100) m`. Randomized: source uniform throughout map. Controlled: source uniform in area within a disk of default radius `75 m` around fixed centre `(500,350) m`; both centre/radius configurable and saved. No convex-hull test. These are benchmark defaults, not simulator constraints. |
| Deployment centre/orientation | Avoid source-aware placement and boundary clipping | **Implemented:** randomized centre uniform in the shared safe square; controlled centre fixed. Shared safe margin `sqrt(max N−1)*100 = 264.575 m`; uniform matched heading `[-π,π)` in both benchmarks. All formations share centroid and `100 m` RMS radius. Reject invalid benchmark configurations (disk/envelope outside map), never outcome-dependent trials. |
| Who owns the estimate? | Only successfully delivered data may enter station inference | **Implemented:** static snapshot fusion; sequential cumulative station history. Current offline measurements are excluded; earlier deliveries remain stored after disconnection, with no backfill. |
| Precisely what each drone knows | “Known drone positions” initial assumption does not imply instantaneous global message access | [GUIDED] knowledge; exact observation [OPEN] |
| DoA angular reference and sensor heading | Consistent interpretation of directed rays | **Implemented:** radians, global +x zero, counterclockwise, wrap to `[-π,π)`; known heading; independent Gaussian noise `σ=2°`. |
| Histories and repeated observation correlation | Valid information accumulation | **Implemented:** independent noise each snapshot; cumulative acquisition-position history for a static source. Sliding windows, buffering/delay, correlation and distributed fusion remain future alternatives. |
| Hard connectivity vs penalty | Separate terminal, continuous and accuracy outcomes | **Implemented:** static all-drone snapshot feasibility unchanged. Sequential outages/reconnections are allowed; final-error Q uses terminal connectivity. No connectivity-dependent move rejection/early exit. Future RL handling remains [OPEN]. |
| Sequential defaults / budgets | Provisional diagnostic choices, not validated UAV/training specifications | **Implemented:** main N=5/T=60; debug N=3/T=30; Δt=2 s, vmax=5 m/s (≤10 m/step). Initial t=0 measurement plus one per move. Default controlled polygon starts, ranges 200/350 m, 8 seeds × 25 scenes; all configurable. No fixed-five-drone/60-step numerical assumptions. |
| Baseline evaluation budget | Avoid sensing/horizon advantages | **Implemented:** static one-snapshot comparisons unchanged. Sequential controllers share initialization, `(T+1)N` generated slots, angular noise, horizon and movement limits; actual travel/delivered counts are separately reported. |
| Policy learning setup | Centralized/independent/CTDE/parameter sharing and variable team-size handling | [OPEN], MARL [REQUIRED] |
| Reward and termination | Determines optimization behavior and prevents degenerate policies | [OPEN] |
| Static test distribution and trial budget | Replication and honest uncertainty | **Implemented:** seeds `11,29,47,71,101,131,173,211`, 125 scenes each, `N=3/5/8`, three ranges, four formations. Balanced setting weights. 2,000 percentile bootstrap replicates of whole scenario IDs (fixed analysis seed), paired across methods and keeping settings together; per-seed variability saved. No training split or map generalization claimed. |
| Realism/integration | Ray tracing/wind/3D changes modeling burden and may obscure the core research | [RESEARCH] unless requested later |

**Rule for Codex:** Keep this living document aligned with verified implementation, preserving its section structure and START / SUBMISSION / EXPAND progression. Present consequential unresolved choices explicitly; record selected choices here, keep the change log concise and leave `SPEC.md` unchanged without approval. Never describe a selected candidate as “what the professors required.”

## 7. What prior work suggests **not** to assume

| Avoid assuming | Reason / evidence | Better boundary |
|---|---|---|
| “Bearing-only cooperative localization under comm constraints is new” | Vander Hook et al. already studied it with mobile bearing sensors and communication limits [L1]. | Evaluate the incremental learned-coordination result; no novelty claim without direct comparison. |
| “Optimal unconstrained geometry is optimal for SWARM-LINK” | Optimal bearings-only formations in [L2] depend on geometry/range assumptions; SWARM-LINK adds reachability. | Evaluate only feasible formations or report violations explicitly. |
| “Better information bound guarantees better realized RMSE” | CRLB is a bound under assumptions, not an estimator-error observation. | Report realized error and theoretical bound separately. |
| “The transmitter location may be supplied to the planner because the simulator knows it” | That turns active localization into an oracle placement problem. | Use truth only for sensing simulation, supervised evaluation, and possibly reward/privileged training quantities with disclosure. |
| “RSS-based RF MARL is a drop-in DoA solution” | The 2020 study [L3] uses RSS, a mobile target and different constraints. | Borrow formulation/experimental ideas, not its sensing equations. |
| “Heterogeneous hierarchical agents are required” | The 2026 RF MARL study [L4] adds AoA+RSS and role/cluster mechanisms. | Treat hierarchy as optional after basic homogeneous DoA work succeeds. |
| “A graph-connected RL policy automatically satisfies permanent connectivity” | Graph architectures can learn behavior but do not inherently guarantee hard invariants [L5]. | Check/enforce reachability at every required step, and report failures. |
| “Wind / full ray tracing must exist before learning begins” | The core Project 5 specification requires DoA/connectivity/MARL, not CFD or full 3D ray tracing. | Keep realism as a later replaceable subsystem. |
| “A centrally controlled concatenated action vector proves decentralized coordination” | Central controller and locally executing agents have different information assumptions. | Describe the executed policy honestly; demonstrate local execution if claimed. |

## 8. Reading map: learn by subsystem, not by paper order

| Topic / component | First read | Follow-up if implementing that part | Scope caveat |
|---|---|---|---|
| **Active sensing, constraints, classical planning** | [L1] Vander Hook et al., 2015 | [L6] Zhou et al., 2024 | [L1] is not a ready-made permanent-ground-station-connectivity implementation. |
| **DoA geometry, FIM/CRLB** | [L2] Wang et al., 2023 | [L6] Zhou et al., 2024 | Optimality conditions differ with source range/unknown position. |
| **Information-aware UAV route planning** | [L7] Chen et al., ICASSP 2026 (Stefano's link) | [L1] | [L7] is not multi-agent connectivity-constrained RL. |
| **RF-source MARL** | [L4] Sun et al., 2026 | [L3] Chen et al., 2020 | [L4] AoA+RSS heterogeneous; [L3] RSS mobile target. |
| **Graph connectivity and multi-agent policies** | [L5] Mishra et al., 2024/2025 | [L8] Zhu et al., survey 2024 | [L5] monitors environments and localizes robots, not an RF transmitter. |

**References** (links are to verified abstracts/full text where available; reading a linked abstract alone does **not** verify all implementation details):

- **[L1]** J. Vander Hook, P. Tokekar, V. Isler, *Algorithms for Cooperative Active Localization of Static Targets With Mobile Bearing Sensors Under Communication Constraints*, IEEE TRO (2015). [DOI](https://doi.org/10.1109/TRO.2015.2432612) · [author PDF](https://josh.vanderhook.info/media/pdf/tro2015cooploc.pdf).
- **[L2]** S. Wang et al., *Optimal Geometry and Motion Coordination for Multisensor Target Tracking with Bearings-Only Measurements*, Sensors (2023). [Open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385148/).
- **[L3]** Y.-J. Chen et al., *Autonomous Tracking Using a Swarm of UAVs: A Constrained Multi-Agent Reinforcement Learning Approach*, IEEE TVT (2020). [DOI](https://doi.org/10.1109/TVT.2020.3023733).
- **[L4]** *Hierarchical Role-Based Multi-Agent Reinforcement Learning for UHF Radiation Source Localization with Heterogeneous UAV Swarms*, Drones (2026). [Open access](https://www.mdpi.com/2504-446X/10/1/54).
- **[L5]** M. Mishra et al., *Multi-Agent Deep Reinforcement Learning for Persistent Monitoring With Sensing, Communication, and Localization Constraints*, IEEE TASE (online 2024, volume 2025). [DOI](https://doi.org/10.1109/TASE.2024.3385412) · [preprint](https://arxiv.org/abs/2109.06831).
- **[L6]** L. Zhou et al., *Robust Multi-UAV Placement Optimization for AOA-Based Cooperative Localization*, IEEE TIV (2024). [DOI](https://doi.org/10.1109/TIV.2024.3362818).
- **[L7]** L. Chen et al., *UAV Path Planning for Radio Frequency Signal Localization via CRLB-Based Uncertainty Minimization*, ICASSP (2026). [DOI](https://doi.org/10.1109/ICASSP55912.2026.11463984). Full technical implementation should be checked against the manuscript, not just its abstract.
- **[L8]** C. Zhu, M. Dastani, S. Wang, *A survey of multi-agent deep reinforcement learning with communication*, Autonomous Agents and Multi-Agent Systems (2024). [Open access](https://link.springer.com/article/10.1007/s10458-023-09633-6).

### Optional integration points (not prerequisites)

- **Project 4 / Alessia Radicula:** synthetic scenes, Sionna radio-propagation/coverage maps, potentially more realistic communication-link or measurement models. Her expertise also directly overlaps RF localization and DoA/triangulation.
- **AIRWISE / Arthur Garbuz:** wind field / flight-cost information and higher-fidelity drone/terrain simulation; potential extension to movement costs, not required for core SWARM-LINK.
- **Stefano Rini:** SWARM-LINK coordination, distributed/ML research direction and scope; **Luca Barletta:** academic scope, information/communication theory, deliverables/evaluation.

**If deeper replication is needed:** obtain the full PDFs for paywalled [L3], [L6], [L7], and any code/supplementary materials. Do not invent pseudocode, hyperparameters or experimental conditions from abstracts. Scientific choices can be narrowed after those sources are read.

### Implementation change log

| Date | Meaningful change | Evidence / remaining scope |
|---|---|---|
| 2026-10-23 | Completed static START experiment; recorded agreed connectivity, ray/LS validity, paired formation fairness and empirical-Q decisions in §§2–6. Added README, lean AGENTS instructions, numerical code, focused tests and reproducible artifacts. | 28 tests pass; two full runs reproduce raw logs exactly; findings/limits in §3.9. SUBMISSION/EXPAND remain documented future stages. |
| 2026-10-23 | Added controlled-v1 beside unchanged randomized-v1; separate reports, source-disk configuration, labelled scene/error/disconnected plots and L rescoring. Documented benchmark/core separation, incremental-comparison principles and postponed line-error study. | 36 tests pass; results/limits in §3.9. No new Python modules, dependencies or future-stage infrastructure. |
| 2026-10-23 | Migrated environment setup to a uv project: `pyproject.toml` replaces `requirements.txt`, `.python-version` pins tested Python 3.14.8, and `uv.lock` locks all dependencies. README uses `uv sync --locked` / `uv run --locked` and explains package inspection without bare pip. No build backend or new dependency. | Locked sync and lock consistency check pass; `uv pip check` reports compatible packages; all 36 tests pass through `uv run --locked`; CLI help works. Verified interpreter is project `.venv` with unchanged NumPy 2.5.4 / Matplotlib 3.11.2. Other supported Python versions and clean-machine Python downloads were not tested. Setup reference: [uv project guide](https://docs.astral.sh/uv/guides/projects/). |
| 2026-10-23 | Added sequential-v1: bounded hold/random motion, cumulative station history, sampled outages/reconnections, separate terminal/continuous outcomes and final-error Q. Existing modules extended; static artifacts preserved. | 50 tests pass; main/debug and repeat-run findings, limits and study pointers in §3.9. RL, deliberate reconnection, wind/terrain and new estimators remain unimplemented. |
