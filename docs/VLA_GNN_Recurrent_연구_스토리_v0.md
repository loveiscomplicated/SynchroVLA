# VLA-GNN-Recurrent 연구 차별점·핵심 목표·전체 스토리 v0

## 1. 핵심 연구 질문

> **큰 VLA가 매번 perception과 action generation을 함께 수행해야 하는가?
> 아니면 semantic reasoning은 느린 pretrained VLM에 맡기고,
> task-relevant geometry와 temporal state를 가진 작은 closed-loop
> controller가 실시간 motor control을 담당하는 편이 더 효율적이고
> 강건한가?**

## 2. 기존 접근에서 겨냥하는 한계

  -----------------------------------------------------------------------------
  한계                    정확한 문제 정의              우리의 대응
  ----------------------- ----------------------------- -----------------------
  **Action chunking**     긴 action sequence를 미리     짧은 주기의 recurrent
                          생성하면 inference 효율은     closed-loop action
                          좋아지지만 실행 중 환경       generation
                          변화에 즉각 반응하기 어렵다.  

  **Stale observation**   Action이 실행될 때 model이    빠른 dynamic graph
                          사용한 visual observation이   update + adaptive
                          이미 과거 상태일 수 있다.     visual re-anchor

  **Scale 의존**          Geometry와 control 관계까지   robot-object 관계와
                          큰 network가 데이터로         Cartesian action에
                          학습하도록 맡긴다.            explicit inductive bias
                                                        부여

  **Appearance / domain   Pixel representation은        task-relevant
  shift**                 texture·lighting·background   geometry를 graph
                          변화에 민감하다.              interface로 추상화
  -----------------------------------------------------------------------------

## 3. 핵심 제안

**Slow semantic reasoning과 fast embodied control을 분리한다.**

Frozen compact VLM은 **"무엇을 해야 하는가"**를 제공하고, Dynamic
Interaction Graph는 **"현재 robot과 object의 관계가 어떠한가"**를
표현한다.

작은 GNN + GRU controller는 이 structured state를 높은 주기로 처리해
즉각적인 action correction을 수행한다.

``` text
Slow:
Image + Language → Frozen VLM → Task semantics
                                  ↓
Fast:
Sensors / Tracker → Dynamic Graph → GNN → GRU → ΔEE Action
                       ↑
             Adaptive visual re-anchor
```

## 4. 차별점

1.  **VLM을 robot controller로 직접 fine-tuning하지 않는다.**\
    Pretrained semantic intelligence로 재사용하고, 학습 대상은
    projector + GNN + recurrent controller 중심으로 제한한다.

2.  **Graph를 단순 추가 feature가 아니라 structured world-state
    interface로 사용한다.**\
    VLM의 semantic reasoning과 실시간 controller 사이에서 robot-object
    관계를 명시적으로 표현한다.

3.  **Recurrence를 실제 시간축의 closed-loop memory로 사용한다.**\
    같은 representation을 inference 내부에서 반복 refinement하는 것이
    아니라, 실제 control step 사이의 상태와 행동 진행을 기억한다.

4.  **Adaptive visual re-anchor를 사용한다.**\
    항상 무거운 vision inference를 수행하지 않고 graph tracking을
    유지하다가, 일정 주기 또는 uncertainty/event가 발생했을 때 정확한
    vision observation으로 상태를 다시 보정한다.

5.  **입력과 출력에 일관된 geometric inductive bias를 준다.**\
    Graph의 relative geometry와 ΔEE action을 동일한 Cartesian 표현
    체계에 둔다.

## 5. 가장 중요한 성능 가설

우리 모델이 **모든 상황에서 대형 VLA보다 우수하다**고 주장하는 것이
목표가 아니다.

핵심 가설은 다음과 같다.

> **실행 도중 현실이 변하고 빠른 correction이 필요한 manipulation에서는,
> 적절한 embodied inductive bias를 가진 compact structured controller가
> 더 큰 chunk-based VLA와 경쟁하거나 능가할 수 있다.**

특히 다음 조건을 중점적으로 평가한다.

-   **Dynamic perturbation:** reaching 도중 target object 이동,
    destination 이동, grasp 직전 pose 변화
-   **Observation delay / latency:** perception 또는 VLA inference
    지연을 인위적으로 증가
-   **Precision / contact:** alignment, handover, stacking 등 작은
    오차가 실패로 이어지는 task
-   **Domain shift:** texture, lighting, background, camera condition
    변화 및 가능하면 sim → real

## 6. Main result가 보여줘야 할 것

  -----------------------------------------------------------------------
  연구 질문                           주요 지표
  ----------------------------------- -----------------------------------
  환경 변화에 더 빨리 반응하는가?     perturbation success, recovery
                                      time, trajectory error

  작은 모델로 경쟁 가능한가?          success rate 대비 trainable
                                      parameters / FLOPs / latency

  stale observation에 강한가?         observation/action delay 증가에
                                      따른 성능 변화

  Graph가 실제로 필요한가?            flat state vs graph, GNN off/on

  Recurrence가 실제로 필요한가?       feed-forward controller vs GRU

  Re-anchor가 필요한가?               every-step vs fixed-N vs adaptive
                                      re-anchor

  Domain-invariant interface인가?     appearance shift 및 sim→real gap
  -----------------------------------------------------------------------

## 7. Sim2Real 주장의 현재 수준

현재 단계에서 **"Graph가 Sim2Real gap을 줄인다"라고 단정하지 않는다.**

가설은 다음과 같다.

> Pixel appearance보다 robot-object relative geometry가 더
> domain-invariant한 control interface가 될 수 있다.

다만 실제 환경에서 object pose / graph 추출 자체가 실패하면 domain gap이
perception 앞단으로 이동할 수 있다. 따라서 **perception error와
controller error를 분리해서 측정**해야 한다.

## 8. 논문 스토리 한 문장

> **Rather than repeatedly using a large VLA to generate temporally
> extended actions, we separate slow semantic reasoning from fast
> embodied control, using a dynamically re-anchored interaction graph
> and a lightweight recurrent controller to achieve responsive
> manipulation with a compact frozen VLM.**

## 9. 현재 범위 밖: 후속 연구

**Skill Composer / concurrent motor primitives는 본 연구의 core
architecture에서 제외한다.**

현재 연구에서는 direct recurrent control의 효과를 먼저 검증한다.

후속 연구에서는 다음과 같이 확장할 수 있다.

``` text
Current:
Graph → Recurrent Controller → Action

Expansion:
Graph → Recurrent Skill Composer
                    ↓
          Concurrent motor primitives
                    ↓
                  Action
```

즉 **graph-conditioned recurrent controller를 hierarchical skill
composer로 확장하는 문제**는 별도의 expansion study 또는 future work로
둔다.
