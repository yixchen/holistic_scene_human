# holistic_scene_human

## Clone this repo
    git clone https://github.com/yixchen/holistic_scene_human.git
The root of this repo will be referenced to `PROJECT_ROOT` in the followings.

## Set up [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) as instructed



## Set up [Lifting from the Deep](https://github.com/DenisTome/Lifting-from-the-Deep-release)

### Install following original repo
- First run `git clone https://github.com/DenisTome/Lifting-from-the-Deep-release.git`
- Run `setup.sh` to retreive the trained models and to install the external utilities.
- Copy `$PROJECT_ROOT/thirdparty/pose_3d.py` to `$Lifting-from-the-Deep-release/applications`
- Change `ROOT_PATH` in `pose_3d.py` to be your data directory