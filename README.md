# 2654 Reinforcement Learning on Underactuated Watercraft

Train a policy in Gazebo, evaluate it, and run it on the real boat. All settings
are in JSON files; commands take no arguments.

## Where things are

| I want to… | Location | Command | Settings |
| --- | --- | --- | --- |
| Train in simulation | `src/watercraft/train.py` | `python -m watercraft.train` | `configs/train.json` |
| Evaluate in simulation | `src/watercraft/evaluate.py` | `python -m watercraft.evaluate` | `configs/evaluate.json` |
| Plot or compare runs | `src/watercraft/results.py` | `watercraft-plot`, `watercraft-compare` | `configs/evaluate.json` (`results`) |
| Export a checkpoint to `.npz` | `src/watercraft/policy.py` | `python -m watercraft.policy` | `configs/evaluate.json` |
| Run a policy on the real boat | `src/watercraft/deploy/` | `python -m watercraft.deploy` | `configs/deploy.json` |
| Calibrate the camera, make markers | `vision_tracking/scripts/` | `python vision_tracking/scripts/<name>.py` | `vision_tracking/config.json` |
| Change the Gym environment | `src/gym_gazebo/envs/pool_gym_env/pool_gym_v0.py` | | |
| Change the simulated world | `src/gym_gazebo/envs/ros_ws/src/boat_gazebo/` | `catkin_make` | |
| Flash the ESP32s | `firmware/transmitter/`, `firmware/receiver/` | `pio run` | `platformio.ini` |

After `pip install -e .` the commands are also available as `watercraft-train`,
`watercraft-evaluate`, `watercraft-export` and `watercraft-deploy`.

```text
RL-Watercraft/
├── configs/
│   ├── train.json            # train, algorithm, task, sim, policy
│   ├── evaluate.json         # evaluate, results (plot/compare)
│   └── deploy.json           # camera, serial, motor calibration, goal, weights
├── src/
│   ├── watercraft/
│   │   ├── train.py          # entry point: train in sim
│   │   ├── evaluate.py       # entry point: evaluate in sim
│   │   ├── results.py        # plotting and run comparison
│   │   ├── deploy/           # entry point: run a policy on the real boat
│   │   │   ├── __main__.py   # camera loop, controller, display
│   │   │   ├── vision.py     # ArUco marker geometry and pose
│   │   │   └── motors.py     # serial send and stop
│   │   ├── algorithms/cem.py # the training loop
│   │   ├── sim_env.py        # applies train.json to the Gazebo environment
│   │   ├── policy.py         # network, loading, export
│   │   ├── contract.py       # observation and action conventions
│   │   ├── config.py         # loads the config files
│   │   └── logging.py        # run folders, CSV, TensorBoard
│   ├── gym_gazebo/           # existing Gym/ROS package, unchanged
│   └── policy_contract.py    # import used by gym_gazebo; re-exports watercraft.contract
├── firmware/                 # transmitter/ and receiver/ PlatformIO projects
├── vision_tracking/
│   ├── config.json
│   ├── calibration/          # camera_calibration.npz, images/
│   ├── markers/              # printable ArUco files
│   └── scripts/              # calibrate_camera, camera_preview, list_cameras, generate_markers
├── tests/
└── docs/
```

`runs/`, `.venv/` and the Catkin `build/` and `devel/` folders are local output
and are not tracked, apart from three historical runs kept under `runs/`.

## Setup

Use one virtual environment (or the system Python) per machine, at the
repository root, never inside a source folder.

```bash
python3 -m pip install --upgrade pip        # editable installs need pip 21.3 or newer
python3 -m pip install -e '.[train,sim]'    # Gazebo machine
python3 -m pip install -e '.[hardware]'     # laptop that drives the boat
```

Without installing, set `PYTHONPATH` to the `src` folder instead:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

ROS, Gazebo and Catkin are system dependencies. Build the workspace once and
source it in every shell that runs the simulator:

```bash
source /opt/ros/noetic/setup.bash
cd src/gym_gazebo/envs/ros_ws && catkin_make
source src/gym_gazebo/envs/ros_ws/devel/setup.bash
```

The hydrodynamics plugins in `ros_ws/src/` (`wave_gazebo_plugins`,
`usv_gazebo_plugins`) were cloned from https://github.com/osrf/vrx/tree/gazebo_classic.

## Train, evaluate and view results

1. Edit `configs/train.json`. The `train` section sets the algorithm, a new run
   name, the seed and the time budget; `algorithm` holds the CEM settings; `task`
   the goal and reward; `sim` the environment options; `policy` the network size
   and observation scales.
2. Run `python -m watercraft.train`. Set `GAZEBO_GUI=false` to run headless.
3. Edit the `evaluate` section of `configs/evaluate.json`: the run and a
   checkpoint filename such as `best.npz`.
4. Run `python -m watercraft.evaluate`.
5. Edit the `results` section of `configs/evaluate.json` to select runs, then run
   `watercraft-plot` or `watercraft-compare`. An empty run list selects every run
   with metrics.

Evaluation and deployment take the policy settings from the `config.json` saved
beside the checkpoint they load, so the observation scales always match the
weights.

Resume training by setting `train.resume` to `runs/<name>/checkpoints/latest.pt`
and keeping the original settings. Set `train.initialize_from` instead to start a
new run from existing `.pth`, `.pt` or `.npz` weights.

To add an algorithm, put its `train(settings, env, directory, resume)` function in
`src/watercraft/algorithms/<name>.py`, add the name to `AVAILABLE` in
`src/watercraft/algorithms/__init__.py`, and select it with `train.algorithm`.

### Output

```text
runs/<name>/
├── config.json                  # fully resolved training settings
├── checkpoints/
│   ├── best.pth / best.npz      # actor used by the best scored collection
│   ├── latest.pt                # learner, optimizer, elite memory and RNG state
│   └── final.pth / final.npz
├── training.csv                 # per-episode metrics
├── iterations.csv               # per-update metrics
├── summary.json
├── tensorboard/
├── graphs/training.png
└── evaluations/sim/<timestamp>.csv/.json/.png
```

Comparisons go to `runs/comparisons/`. `best` is chosen from exploratory training
episodes, not an independent validation score; use evaluation for frozen-policy
results.

## Run on the real boat

1. Edit `configs/deploy.json`: camera index, serial port, calibration file, goal,
   motor calibration, and `policy_weights` (a `.npz` from a run folder).
   `controller` is `policy` or `p` (the proportional controller).
2. Leave `"arm": false` for a dry run that tracks and displays commands without
   opening the serial port. Set it to `true` to drive the motors.
3. Run `python -m watercraft.deploy`. Keep all four corner markers visible; `R`
   recalibrates, `Q` or `Esc` quits. Motors stop when the marker is lost and on exit.

Camera tools read `vision_tracking/config.json` and the camera settings in
`configs/deploy.json`:

```bash
python vision_tracking/scripts/list_cameras.py
python vision_tracking/scripts/camera_preview.py
python vision_tracking/scripts/calibrate_camera.py
python vision_tracking/scripts/generate_markers.py
```

## Tests and diagnostics

```bash
python -m unittest discover -s tests -v     # no ROS, camera or serial port needed
python tests/sim_diagnostic.py              # random actions in Gazebo
python tests/reset_diagnostic.py            # checks that the world resets
```

## Original installation notes (Ubuntu 18.04, ROS Melodic)

These notes come from the gym-gazebo project this repository started from. The
current simulation machine runs ROS Noetic.

#### Basic requirements
- ROS Melodic: Desktop-Full Install recommended, includes Gazebo 9.0.0 (http://wiki.ros.org/melodic/Installation/Ubuntu).
- Gazebo 9.0.0

#### ROS Melodic related dependencies
```
sudo apt-get install \
python-pip python3-vcstool python3-pyqt4 \
pyqt5-dev-tools \
libbluetooth-dev libspnav-dev \
pyqt4-dev-tools libcwiid-dev \
cmake gcc g++ qt4-qmake libqt4-dev \
libusb-dev libftdi-dev \
python3-defusedxml python3-vcstool \
ros-melodic-octomap-msgs        \
ros-melodic-joy                 \
ros-melodic-geodesy             \
ros-melodic-octomap-ros         \
ros-melodic-control-toolbox     \
ros-melodic-pluginlib	       \
ros-melodic-trajectory-msgs     \
ros-melodic-control-msgs	       \
ros-melodic-std-srvs 	       \
ros-melodic-nodelet	       \
ros-melodic-urdf		       \
ros-melodic-rviz		       \
ros-melodic-kdl-conversions     \
ros-melodic-eigen-conversions   \
ros-melodic-tf2-sensor-msgs     \
ros-melodic-pcl-ros \
ros-melodic-navigation \
ros-melodic-sophus
```

#### Install Python Packages:
```
sudo pip install gym
sudo apt-get install python-skimage
sudo pip install h5py
pip install tensorflow-gpu (if you have a gpu if not then just pip install tensorflow)
sudo pip install keras
```
