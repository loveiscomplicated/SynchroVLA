# VLA-GNN-Recurrent 모델 설계 v0

> **핵심 방향:** Slow semantic reasoning + dynamic graph state + fast
> recurrent closed-loop control

## 1. 설계 목표

무거운 VLM이 매 제어 시점마다 직접 action을 생성하지 않도록 한다. VLM은
**task semantics**를 제공하고, 빠른 제어는 **graph-conditioned recurrent
controller**가 담당한다.

핵심은 **semantic reasoning과 embodied control의 시간축을 분리**하는
것이다.

## 2. 전체 아키텍처

``` text
RGB + Language
      ↓
Frozen compact VLM
(우선 후보: SmolVLM2-500M)
      ↓
Trainable semantic projector
      ↓
Task node / semantic context
      │
Visual re-anchor ──→ Dynamic Interaction Graph ←── proprioception / FK / tracker
                           ↓
                     Edge-aware GNN
                           ↓
                   Recurrent Controller
                           ↓
             Δ End-Effector pose + gripper
                           ↓
                    Constrained IK
                           ↓
                     Joint command
```

## 3. 모듈별 v0 선택

  ------------------------------------------------------------------------------------------
  모듈              v0 선택             역할                   핵심 이유
  ----------------- ------------------- ---------------------- -----------------------------
  VLM               **SmolVLM2-500M,    명령·장면의 semantic   compact baseline과 비교가
                    frozen**            context                쉽고 control 학습과 분리 가능

  Semantic          **2-layer MLP**     VLM feature를 작은     VLM은 freeze하면서 downstream
  projector                             task context로 변환    표현은 적응

  Vision → Graph    **GroundingDINO +   중요 object의          새 vision-to-graph 모델을
                    SAM2 + RGB-D**;     위치·pose·confidence   처음부터 학습하는 위험 회피
                    정밀 pose 필요 시   추출                   
                    FoundationPose                             

  GNN               **3-layer           robot-object 관계      edge의 relative geometry가
                    edge-aware MPNN,    message passing        핵심 정보
                    hidden 128--256**                          

  Recurrent         **2-layer GRU,      행동의 temporal        작고 빠르며 필요한 memory
  controller        hidden 256--512**   continuity와 빠른      기능에 충분
                                        correction             

  Action head       **ΔEE pose +        Cartesian 공간에서     graph 입력의 geometry와
                    gripper**           다음 작은 움직임 출력  action space 정렬

  Low-level control **Constrained IK**  EE 목표를 joint        joint
                                        command로 변환         limit·singularity·preferred
                                                               posture를 명시적으로 처리
  ------------------------------------------------------------------------------------------

## 4. Dynamic Interaction Graph

노드는 크게 **robot / object / task**의 세 종류로 구성한다.

-   **Robot nodes:** 양팔의 주요 joint, end-effector, gripper. Joint/EE
    pose는 proprioception과 forward kinematics로 매 제어 시점 갱신한다.
-   **Object nodes:** task-relevant object와 destination/target. 위치,
    orientation, tracking confidence 등을 저장한다.
-   **Task node:** frozen VLM의 semantic feature를 projector로 축소해
    표현한다. Task instruction과 현재 중요한 entity 정보를 전달한다.

## 5. Task node와 edge 정의

Task node를 모든 joint에 완전 연결하는 것보다는, **의미적으로 직접
관련된 object와 end-effector/gripper에 연결하는 것**을 기본안으로 둔다.
Joint는 기존 kinematic edge를 통해 task 정보를 전달받는다.

  -----------------------------------------------------------------------
  Edge 종류               예시                    Feature / relation
  ----------------------- ----------------------- -----------------------
  Kinematic               shoulder ↔ elbow ↔      kinematic adjacency,
                          wrist ↔ EE              relative pose

  Spatial                 R_EE ↔ cup              Δxyz, distance,
                                                  relative orientation

  Interaction             gripper ↔ cup           contact, grasp state,
                                                  confidence

  Task-semantic           Task ↔ cup              target_object

  Task-semantic           Task ↔ bowl             destination

  Task-semantic           Task ↔ R_EE             active/relevant
                                                  effector
  -----------------------------------------------------------------------

### Task conditioning ablation 후보

1.  **Task → object/EE만 연결**
2.  **Task → 모든 node 연결**
3.  Graph pooling 이후 task embedding을 concat

## 6. Visual re-anchor와 stale-state 제어

최종 지향점은 매 step RGB를 다시 graph로 만드는 방식이 아니라, **빠른
graph tracking + 정확한 visual re-anchor**를 결합하는 것이다.

### 비교할 세 방식

1.  **Every-step re-anchor**\
    매 control step vision → graph 재구성. 정확하지만 계산 비용이 높다.

2.  **Fixed-N re-anchor**\
    N step마다 vision → graph refresh. 저렴하지만 compounding error
    위험이 있다.

3.  **Adaptive re-anchor --- Main**\
    Graph state는 proprioception/tracker로 계속 갱신한다. 일정 주기 또는
    uncertainty, contact, tracking failure, grasp/release 등의 event가
    발생하면 vision으로 다시 보정한다.

Re-anchor 시에는 graph뿐 아니라 **recurrent hidden state 역시 최신
observation에 맞게 correction**하는 방법을 고려한다.

## 7. Δ End-Effector Action

Controller는 각 관절의 delta를 직접 예측하지 않고, 각 end-effector가
Cartesian 공간에서 얼마나 이동·회전해야 하는지를 예측한다.

``` text
Left  : Δx, Δy, Δz, Δrotation, gripper
Right : Δx, Δy, Δz, Δrotation, gripper

Network → ΔEE target → constrained IK → joint commands
```

이 선택의 핵심은 **입력 graph의 상대 위치 관계와 출력 action이 같은
geometric language를 사용한다는 점**이다.

네트워크는

> 컵이 손보다 +x 방향에 있다 → 손을 +x 방향으로 이동

을 학습하고, 구체적인 관절 변화는 IK에 맡긴다.

## 8. 아직 고정하지 않은 설계 변수

-   Object pose 표현: xyz만 사용할지, 6D pose까지 사용할지
-   Graph temporal feature: velocity를 직접 넣을지, GNN 자체에 짧은
    history를 줄지
-   Re-anchor trigger: fixed interval + uncertainty threshold의 구체적
    정의
-   GRU hidden-state correction 방식
-   Action rotation representation과 control frequency
-   **MPNN vs EGNN**, **GRU vs SSM/Mamba**는 후속 ablation 후보
