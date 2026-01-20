export LIBERO_CONFIG_PATH=$PWD
export PYTHONPATH=$PYTHONPATH:$PWD

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export XDG_RUNTIME_DIR=/tmp
export NVIDIA_DRIVER_CAPABILITIES=all
unset DISPLAY
export MUJOCO_EGL_DEVICE_ID=0


TASK_SUITE=$1
SAVE_NAME=$2

python main.py --args.task_suite_name=${TASK_SUITE} --args.save_name=${SAVE_NAME} --args.save_videos