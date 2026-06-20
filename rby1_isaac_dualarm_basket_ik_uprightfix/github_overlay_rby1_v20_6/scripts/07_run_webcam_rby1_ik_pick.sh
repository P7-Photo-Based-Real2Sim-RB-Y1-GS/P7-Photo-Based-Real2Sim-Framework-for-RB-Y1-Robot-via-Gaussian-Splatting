#!/usr/bin/env bash
set -euo pipefail

PROJECT="${RBY1_PROJECT:-$HOME/rby1_ros2_hand_teleop}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_DEVICE="${CAMERA_DEVICE:-}"
LIST_CAMERAS=0
PROBE_CAMERAS=0
UDP_PORT="${UDP_PORT:-5005}"
WEBCAM_ARGS=()
ISAAC_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --camera-index)
      CAMERA_INDEX="$2"; shift 2 ;;
    --camera-device)
      CAMERA_DEVICE="$2"; shift 2 ;;
    --list-cameras)
      LIST_CAMERAS=1; shift ;;
    --probe-cameras)
      PROBE_CAMERAS=1; shift ;;
    --udp-port)
      UDP_PORT="$2"; shift 2 ;;
    --webcam-no-topmost)
      WEBCAM_ARGS+=(--no-topmost); shift ;;
    --webcam-window-x)
      WEBCAM_ARGS+=(--window-x "$2"); shift 2 ;;
    --webcam-window-y)
      WEBCAM_ARGS+=(--window-y "$2"); shift 2 ;;
    --webcam-width)
      WEBCAM_ARGS+=(--width "$2"); shift 2 ;;
    --webcam-height)
      WEBCAM_ARGS+=(--height "$2"); shift 2 ;;
    *)
      ISAAC_ARGS+=("$1"); shift ;;
  esac
done

CAM_SCRIPT="$PROJECT/camera/webcam_holistic_rby1_udp.py"
ISAAC_SCRIPT="$PROJECT/isaac/run_webcam_rby1_ik_table_pick.py"

if [[ ! -f "$CAM_SCRIPT" || ! -f "$ISAAC_SCRIPT" ]]; then
  echo "[ERROR] IK patch files are not installed under $PROJECT" >&2
  echo "        Run apply_rby1_webcam_ik_patch.sh first." >&2
  exit 1
fi

choose_webcam_python() {
  local candidates=()
  [[ -n "${WEBCAM_PYTHON:-}" ]] && candidates+=("$WEBCAM_PYTHON")
  candidates+=("/usr/bin/python3" "python3")
  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      if "$candidate" -c 'import cv2, mediapipe, numpy' >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if ! CAM_PYTHON="$(choose_webcam_python)"; then
  echo "[ERROR] cv2/mediapipe/numpy가 설치된 Python을 찾지 못했습니다." >&2
  echo "        /usr/bin/python3 -m pip install --break-system-packages opencv-python mediapipe numpy" >&2
  exit 1
fi

if [[ "$PROBE_CAMERAS" -eq 1 ]]; then
  "$CAM_PYTHON" "$PROJECT/tools/probe_rby1_camera.py"
  exit $?
fi

if [[ "$LIST_CAMERAS" -eq 1 ]]; then
  "$CAM_PYTHON" "$CAM_SCRIPT" --list-cameras
  exit $?
fi

CAM_CMD=("$CAM_PYTHON" "$CAM_SCRIPT" --camera-index "$CAMERA_INDEX" --udp-port "$UDP_PORT")
if [[ -n "$CAMERA_DEVICE" ]]; then
  CAM_CMD+=(--camera-device "$CAMERA_DEVICE")
fi
CAM_CMD+=("${WEBCAM_ARGS[@]}")
"${CAM_CMD[@]}" &
CAM_PID=$!

cleanup() {
  kill "$CAM_PID" >/dev/null 2>&1 || true
  wait "$CAM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
sleep 1
if ! kill -0 "$CAM_PID" >/dev/null 2>&1; then
  echo "[ERROR] Webcam process exited before Isaac Sim startup." >&2
  wait "$CAM_PID" || true
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES

if [[ -n "${ISAACLAB_DIR:-}" && -x "$ISAACLAB_DIR/isaaclab.sh" ]]; then
  "$ISAACLAB_DIR/isaaclab.sh" -p "$ISAAC_SCRIPT" --udp-port "$UDP_PORT" "${ISAAC_ARGS[@]}"
elif [[ -x "$HOME/IsaacLab/isaaclab.sh" ]]; then
  "$HOME/IsaacLab/isaaclab.sh" -p "$ISAAC_SCRIPT" --udp-port "$UDP_PORT" "${ISAAC_ARGS[@]}"
else
  python "$ISAAC_SCRIPT" --udp-port "$UDP_PORT" "${ISAAC_ARGS[@]}"
fi
