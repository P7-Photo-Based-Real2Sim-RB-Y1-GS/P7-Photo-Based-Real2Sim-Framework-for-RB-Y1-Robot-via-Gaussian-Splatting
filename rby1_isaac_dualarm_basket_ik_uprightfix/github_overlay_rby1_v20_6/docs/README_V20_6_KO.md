RB-Y1 v20.6 - 팔꿈치 Jacobian broadcast 오류 수정
===================================================

발생 오류
---------
ValueError:
  operands could not be broadcast together with shapes (7,) (3,) (7,)

원인
----
다음과 같이 3차원 Jacobian 배열에 관절 index 배열을 한 번에 적용하면:

  jacobians[body_index, :3, joint_indices]

NumPy advanced indexing 규칙 때문에 결과가 (3,7)이 아니라
(7,3)으로 바뀔 수 있습니다.

기존 코드는 J_elbow[2]를 사용했기 때문에 길이 3인 값이 나왔고,
7관절 보정 벡터와 더할 수 없었습니다.

수정
----
- body Jacobian을 먼저 2차원 배열로 선택
- 그 다음 관절 column을 선택
- 모든 EE/팔꿈치 Jacobian을 6x7 형식으로 정규화
- transposed Jacobian(dof x 6)도 자동 교정
- elbow Z Jacobian은 항상 J_linear[2, :]의 7개 값 사용
- shape가 잘못되면 의미 있는 진단 오류 출력

설치
----
cd ~/다운로드
unzip -o rby1_webcam_stl_grasp_only_v20_6_jacobian_fix.zip
cd rby1_webcam_stl_grasp_only_v20_6_jacobian_fix

chmod +x apply_rby1_webcam_ik_patch.sh
./apply_rby1_webcam_ik_patch.sh

실행
----
cd ~/rby1_ros2_hand_teleop
conda activate env_isaaclab
export ISAACLAB_DIR=$HOME/IsaacLab

./scripts/07_run_webcam_rby1_ik_pick.sh           --camera-device "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"           --control-side right           --teleop-mode joint           --visibility 0.15

정상 로그
---------
[INFO] v20.6 Jacobian convention:
       spatial_rows x selected_arm_joints = 6x7

이후 기존 자동 접근·그립이 계속 진행되어야 합니다.
