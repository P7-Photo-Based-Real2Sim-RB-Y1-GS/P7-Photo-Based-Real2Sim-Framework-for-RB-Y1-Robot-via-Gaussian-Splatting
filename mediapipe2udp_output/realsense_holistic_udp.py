import cv2
import mediapipe as mp
import pyrealsense2 as rs
import socket
import json
import time
import math
import numpy as np

# =========================================================
# UDP 설정
# =========================================================
# Isaac Sim이 같은 PC에서 실행 중이면 127.0.0.1
# 다른 PC에서 실행 중이면 Isaac Sim PC의 IP 주소로 변경
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# =========================================================
# RealSense 설정
# =========================================================
WIDTH = 640
HEIGHT = 480
FPS = 30

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

profile = pipeline.start(config)

# depth frame을 color frame 기준으로 정렬
align_to = rs.stream.color
align = rs.align(align_to)

# depth scale 확인
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth scale:", depth_scale)

# color camera intrinsics 가져오기
color_stream = profile.get_stream(rs.stream.color)
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

print("Color intrinsics:")
print(color_intrinsics)

# =========================================================
# MediaPipe Holistic 설정
# =========================================================
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================================================
# 추적할 Holistic Pose landmark 번호
# =========================================================
# 그림 기준:
# 20 - right index
# 18 - right pinky
# 16 - right wrist
# 22 - right thumb
# 14 - right elbow
# 12 - right shoulder
# 11 - left shoulder
# 13 - left elbow
# 15 - left wrist
# 17 - left pinky
# 19 - left index
# 21 - left thumb

TARGET_POSE_LANDMARKS = {
    20: "RIGHT_INDEX",
    18: "RIGHT_PINKY",
    16: "RIGHT_WRIST",
    22: "RIGHT_THUMB",
    14: "RIGHT_ELBOW",
    12: "RIGHT_SHOULDER",
    11: "LEFT_SHOULDER",
    13: "LEFT_ELBOW",
    15: "LEFT_WRIST",
    17: "LEFT_PINKY",
    19: "LEFT_INDEX",
    21: "LEFT_THUMB",
}

# =========================================================
# 추가로 추적할 손 landmark 번호
# =========================================================
# MediaPipe Hand landmark 기준:
# 4 = THUMB_TIP

TARGET_HAND_LANDMARKS = {
    4: "THUMB_TIP",
}

# =========================================================
# 화면 표시 색상
# OpenCV BGR 순서
# =========================================================
COLOR_POSE_POINT = (0, 0, 255)       # 빨강
COLOR_POSE_TEXT = (0, 255, 0)        # 초록
COLOR_HAND_POINT = (0, 255, 255)     # 노랑
COLOR_HAND_TEXT = (0, 255, 255)      # 노랑
COLOR_COORD = (255, 0, 0)            # 파랑
COLOR_ERROR = (0, 0, 255)            # 빨강

# =========================================================
# 보조 함수
# =========================================================
def get_valid_depth(depth_frame, u, v, search_radius=5):
  
    depths = []

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            px = u + dx
            py = v + dy

            if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
                continue

            d = depth_frame.get_distance(px, py)

            if d > 0 and not math.isnan(d):
                depths.append(d)

    if len(depths) == 0:
        return None

    return sum(depths) / len(depths)


def pixel_to_camera_3d(u, v, depth_m):
    """
    pixel 좌표와 depth를 RealSense 카메라 기준 3D 좌표로 변환.
    반환 좌표 단위: meter

    RealSense camera coordinate:
        X: 카메라 기준 오른쪽 +
        Y: 카메라 기준 아래쪽 +
        Z: 카메라 기준 전방 +
    """
    point_3d = rs.rs2_deproject_pixel_to_point(
        color_intrinsics,
        [float(u), float(v)],
        float(depth_m)
    )

    return {
        "x": point_3d[0],
        "y": point_3d[1],
        "z": point_3d[2],
    }


def convert_camera_to_isaac_position(camera_pos):
    """
    RealSense 카메라 좌표를 Isaac Sim 입력용 좌표로 변환.

    현재 변환:
        RealSense X 오른쪽  -> Isaac X
        RealSense Z 전방    -> Isaac Y
        RealSense Y 아래쪽  -> Isaac Z의 음수

        카메라 오른쪽 = Isaac +X
        카메라 앞쪽   = Isaac +Y
        카메라 위쪽   = Isaac +Z

    Isaac Sim 쪽 좌표계가 다르면 이 함수만 수정
    """
    x_cam = camera_pos["x"]
    y_cam = camera_pos["y"]
    z_cam = camera_pos["z"]

    return {
        "x": x_cam,
        "y": z_cam,
        "z": -y_cam,
    }


def make_3d_landmark_data(
    landmark_id,
    landmark_name,
    image_x,
    image_y,
    image_z,
    depth_frame,
    image_width,
    image_height,
    extra_fields=None,
):


    u = int(image_x * image_width)
    v = int(image_y * image_height)

    u = max(0, min(image_width - 1, u))
    v = max(0, min(image_height - 1, v))

    depth_m = get_valid_depth(depth_frame, u, v, search_radius=5)

    if depth_m is None:
        return None, u, v

    camera_position = pixel_to_camera_3d(u, v, depth_m)
    isaac_position = convert_camera_to_isaac_position(camera_position)

    landmark_data = {
        "landmark_id": landmark_id,
        "landmark_name": landmark_name,

        "image_landmark": {
            "x": image_x,
            "y": image_y,
            "z": image_z,
        },

        "pixel": {
            "u": u,
            "v": v,
        },

        "depth_m": depth_m,

        # RealSense 카메라 기준 실제 3D 좌표, meter 단위
        "camera_position_m": camera_position,

        # Isaac Sim 입력용 좌표, meter 단위
        "position": isaac_position,
    }

    if extra_fields is not None:
        landmark_data.update(extra_fields)

    return landmark_data, u, v


# =========================================================
# Main loop
# =========================================================
print(f"Sending RealSense + MediaPipe Holistic data to udp://{UDP_IP}:{UDP_PORT}")
print("Using MediaPipe Holistic")
print("Pose landmarks:", list(TARGET_POSE_LANDMARKS.keys()))
print("Hand landmarks:", TARGET_HAND_LANDMARKS)
print("Press q to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()

        # depth frame을 color frame 기준으로 정렬
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # MediaPipe 입력은 RGB
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 성능을 위해 writeable 끄기
        rgb_image.flags.writeable = False
        result = holistic.process(rgb_image)
        rgb_image.flags.writeable = True

        h, w, _ = color_image.shape

        detected_pose_landmarks = []
        detected_hand_landmarks = []

        # =========================================================
        # 1. Holistic pose_landmarks 처리
        #    20, 18, 16, 22, 14, 12, 11, 13, 15, 17, 19, 21
        # =========================================================
        if result.pose_landmarks:
            for landmark_id, landmark_name in TARGET_POSE_LANDMARKS.items():
                landmark = result.pose_landmarks.landmark[landmark_id]

                image_x = landmark.x
                image_y = landmark.y
                image_z = landmark.z
                visibility = landmark.visibility

                landmark_data, u, v = make_3d_landmark_data(
                    landmark_id=landmark_id,
                    landmark_name=landmark_name,
                    image_x=image_x,
                    image_y=image_y,
                    image_z=image_z,
                    depth_frame=depth_frame,
                    image_width=w,
                    image_height=h,
                    extra_fields={
                        "source_type": "pose_landmark",
                        "visibility": visibility,
                    },
                )

                if landmark_data is None:
                    cv2.circle(color_image, (u, v), 6, COLOR_ERROR, -1)
                    cv2.putText(
                        color_image,
                        f"POSE {landmark_id}:{landmark_name} depth invalid",
                        (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        COLOR_ERROR,
                        1,
                    )
                    continue

                # visibility도 image_landmark 안에 추가
                landmark_data["image_landmark"]["visibility"] = visibility

                detected_pose_landmarks.append(landmark_data)

                # 화면 표시
                cam = landmark_data["camera_position_m"]

                cv2.circle(color_image, (u, v), 7, COLOR_POSE_POINT, -1)

                cv2.putText(
                    color_image,
                    f"POSE {landmark_id}:{landmark_name}",
                    (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    COLOR_POSE_TEXT,
                    1,
                )

                cv2.putText(
                    color_image,
                    f"x:{cam['x']:.2f} y:{cam['y']:.2f} z:{cam['z']:.2f}m",
                    (u + 10, v + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    COLOR_COORD,
                    1,
                )

            # Pose skeleton 전체 표시
            mp_draw.draw_landmarks(
                color_image,
                result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        # =========================================================
        # 2. Holistic hand_landmarks 처리
        #    엄지 손가락 끝 = hand landmark 4번 THUMB_TIP
        # =========================================================
        hand_sources = [
            ("Left", result.left_hand_landmarks),
            ("Right", result.right_hand_landmarks),
        ]

        for handedness, hand_landmarks in hand_sources:
            if hand_landmarks is None:
                continue

            # 손 skeleton 표시
            mp_draw.draw_landmarks(
                color_image,
                hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

            for landmark_id, landmark_name in TARGET_HAND_LANDMARKS.items():
                landmark = hand_landmarks.landmark[landmark_id]

                image_x = landmark.x
                image_y = landmark.y
                image_z = landmark.z

                landmark_data, u, v = make_3d_landmark_data(
                    landmark_id=landmark_id,
                    landmark_name=landmark_name,
                    image_x=image_x,
                    image_y=image_y,
                    image_z=image_z,
                    depth_frame=depth_frame,
                    image_width=w,
                    image_height=h,
                    extra_fields={
                        "source_type": "hand_landmark",
                        "handedness": handedness,
                    },
                )

                if landmark_data is None:
                    cv2.circle(color_image, (u, v), 6, COLOR_ERROR, -1)
                    cv2.putText(
                        color_image,
                        f"{handedness} HAND {landmark_id}:{landmark_name} depth invalid",
                        (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        COLOR_ERROR,
                        1,
                    )
                    continue

                detected_hand_landmarks.append(landmark_data)

                # 화면 표시
                cam = landmark_data["camera_position_m"]

                cv2.circle(color_image, (u, v), 8, COLOR_HAND_POINT, -1)

                cv2.putText(
                    color_image,
                    f"{handedness} HAND {landmark_id}:{landmark_name}",
                    (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    COLOR_HAND_TEXT,
                    1,
                )

                cv2.putText(
                    color_image,
                    f"x:{cam['x']:.2f} y:{cam['y']:.2f} z:{cam['z']:.2f}m",
                    (u + 10, v + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    COLOR_COORD,
                    1,
                )

        # =========================================================
        # 3. UDP 전송
        # =========================================================
        if detected_pose_landmarks or detected_hand_landmarks:
            data = {
                "timestamp": time.time(),
                "source": "realsense_mediapipe_holistic",
                "coordinate_type": "realsense_camera_3d",
                "unit": "meter",

                "target_pose_landmarks": list(TARGET_POSE_LANDMARKS.keys()),
                "num_pose_landmarks": len(detected_pose_landmarks),
                "pose_landmarks": detected_pose_landmarks,

                "target_hand_landmarks": list(TARGET_HAND_LANDMARKS.keys()),
                "num_hand_landmarks": len(detected_hand_landmarks),
                "hand_landmarks": detected_hand_landmarks,
            }

            message = json.dumps(data)
            sock.sendto(message.encode("utf-8"), (UDP_IP, UDP_PORT))

            print("-----")

            for lm in detected_pose_landmarks:
                cam = lm["camera_position_m"]
                pos = lm["position"]

                print(
                    f"POSE {lm['landmark_id']} {lm['landmark_name']} | "
                    f"cam=({cam['x']:.3f}, {cam['y']:.3f}, {cam['z']:.3f}) m | "
                    f"isaac=({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f}) | "
                    f"vis={lm['image_landmark']['visibility']:.2f}"
                )

            for lm in detected_hand_landmarks:
                cam = lm["camera_position_m"]
                pos = lm["position"]

                print(
                    f"{lm['handedness']} HAND {lm['landmark_id']} {lm['landmark_name']} | "
                    f"cam=({cam['x']:.3f}, {cam['y']:.3f}, {cam['z']:.3f}) m | "
                    f"isaac=({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})"
                )

        cv2.imshow("RealSense + MediaPipe Holistic Sender", color_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    holistic.close()
    pipeline.stop()
    sock.close()
    cv2.destroyAllWindows()
