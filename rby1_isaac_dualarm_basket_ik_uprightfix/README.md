# RBY1 Isaac Sim / Isaac Lab Bridge + Dual-Arm Basket Demo

이 폴더는 업로드한 준비된 USD 파일을 그대로 이용해서 Isaac Lab에서 Isaac Sim을 띄우고,
RBY1 articulation을 자동으로 찾은 뒤 다음 데모를 실행할 수 있게 만든 프로젝트다.

## 포함된 것
- `assets/RBY1_A_v1_0.usd`
- `assets/RBY1_A_v1_1.usd`
- `run_inspect_and_fix_root.py`
- `run_effort_pid_demo.py`
- `run_dual_arm_basket_demo.py`

## 먼저 root 확인
```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /absolute/path/to/rby1_isaac_dualarm_basket/run_inspect_and_fix_root.py --asset v1_1
```

## 기존 오른팔 PID 데모
```bash
./isaaclab.sh -p /absolute/path/to/rby1_isaac_dualarm_basket/run_effort_pid_demo.py --asset v1_1
```

## 새 dual-arm basket 데모
```bash
./isaaclab.sh -p /absolute/path/to/rby1_isaac_dualarm_basket/run_dual_arm_basket_demo.py --asset v1_1
```

반복 실행하려면:
```bash
./isaaclab.sh -p /absolute/path/to/rby1_isaac_dualarm_basket/run_dual_arm_basket_demo.py --asset v1_1 --loop
```

## 데모 내용
이 스크립트는 joint-space scripted state machine으로 다음 순서를 실행한다.

1. 양팔 ready pose
2. 오른팔은 assist pose로 이동
3. 왼팔 pre-grasp pose
4. 왼손 gripper close
5. 왼팔 lift
6. 바구니 위 pose로 이동
7. 왼손 gripper open
8. ready pose 복귀

## 주의
- 이 버전은 **IK/모션플래닝 없이**, MuJoCo/URDF에서 흔히 쓰는 RBY1 joint 이름을 가정한 joint-space 데모다.
- 기본적으로 `right_arm_0~6`, `left_arm_0~6`, `gripper_finger_l1/l2`, `gripper_finger_r1/r2`를 찾는다.
- USD 안 joint 이름이 다르면, 실행 시작 시 출력되는 `Available DOF names`를 보고 `rby1_isaac_connected/common.py`의 이름 리스트만 수정하면 된다.
- scene 안의 실제 object grasp/contact 품질은 USD의 collision/physics 설정에 따라 달라질 수 있다.


## IK demo

Run the new left-arm IK demo with:

```bash
./isaaclab.sh -p /absolute/path/to/run_dual_arm_basket_ik_demo.py --asset v1_1
```

Notes:
- This version uses **Jacobian-based damped-least-squares position IK** for the left arm and keeps the right arm in a safe assist pose.
- The stock RBY1 URDF in the SDK exposes the finger joints as fixed joints, so the mug is carried with a **kinematic attach** during the carry phase instead of a physically actuated finger grasp.
- This is intended as a collision-avoidance-friendly demo scaffold you can refine into full pose IK or motion planning later.


## Drop-target fix
This version removes the release-time teleport to the basket root and instead computes a basket drop anchor from the basket world-space bounding box top center.
