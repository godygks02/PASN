"""결정 사항이 실제 기본값에 반영돼 있는지 감사한다.

연구일지 실험 1–11 과 P0.1–P0.3 에서 내린 결정을, **코드가 실제로 주는 값**과 하나씩
대조한다. 주석이나 노트가 아니라 ``ConvertConfig()`` / ``build_mbe_pasn`` 의 시그니처를
읽는다 -- GPU 시간을 쓰기 전에 "노트에는 그렇게 적혀 있는데 코드는 아니었다" 를 없애기
위한 것.

사용법::

    python experiments/audit_defaults.py
"""
from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mbe import convert as cv  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn, rule_budget  # noqa: E402
from mbe.neuron import MBEConfig  # noqa: E402

cfg = cv.ConvertConfig()
bp = inspect.signature(build_mbe_pasn).parameters
mc = MBEConfig()

# (결정, 출처, 실제값, 기대값)
CHECKS = [
    ("뱅크별 (N,T) 예산 규칙", "실험 1", cfg.pasn_budget, "rule"),
    ("near-zero 부호분할", "실험 2", cfg.pasn_near0, "signed"),
    ("빌더 기본 near0", "실험 2/3", bp["near0"].default, "signed"),
    ("활성함수 디코더 차수 2", "실험 6", cfg.pasn_readout_order, 2),
    ("커널은 학습 (고정 아님)", "실험 6/9", mc.learn_tau, True),
    ("발화율 페널티 off", "실험 8/9", bp["spike_lambda"].default, 0.0),
    ("항등원 tied", "실험 4/10", cfg.pasn_id_tied, True),
    ("항등원 디코더는 차수 1", "실험 6", mc.readout_order, 1),
    ("β off", "P0.2", cfg.pasn_beta, {}),
    ("항등원 signed 도메인 off", "P0.2", cfg.pasn_id_signed, False),
    ("alpha_init auto", "P0.3", cfg.pasn_alpha_init, "auto"),
    ("도달불가 뱅크 공유", "실험 11", "fill_unreachable" in
     inspect.getsource(build_mbe_pasn), True),
]

# 결정이 "열어둔" 것 -- 기본값이 있지만 아직 확정이 아니다.
OPEN = [
    ("항등원 상대오차 타깃", cfg.pasn_id_target, "relative",
     "P0.3 에서 forward-error 기준으로는 0.81× 손해. 실험 10 은 곱 상대오차가 "
     "절대타깃에서 15–49% 라고 측정. **어느 쪽이 옳은지는 Δppl 이 정한다 (P0.4)**"),
    ("항등원 동작점 r", cfg.pasn_id_target_rel, 1e-2,
     "네트워크 전체 동작점의 주 손잡이. P0.4 에서 프론티어로 훑어야 한다"),
    ("라우터 깊이 e_min / id_e_min", (cfg.pasn_e_min, cfg.pasn_id_e_min),
     (-3, -6), "토이에서 정한 값. 실 모델 활성 분포에서 재확인 필요"),
]


def main():
    ok = True
    print("# 결정 → 코드 반영 감사\n")
    print("| 결정 | 출처 | 코드 값 | 기대 | |")
    print("|---|---|---|---|---|")
    for name, src, got, want in CHECKS:
        good = got == want
        ok &= good
        print(f"| {name} | {src} | `{got}` | `{want}` | "
              f"{'✅' if good else '❌'} |")

    print("\n## 아직 열려 있는 것 (기본값은 있지만 확정 아님)\n")
    for name, got, dflt, why in OPEN:
        print(f"- **{name}** = `{got}` — {why}")

    # 규칙 상수가 차수별로 실제로 갈리는지 (실험 7)
    print("\n## 예산 규칙이 디코더 차수를 반영하는가 (실험 7)\n")
    print("| 차수 | 커널 | Δ=1 ε=1e−5 → (N,T) |")
    print("|---|---|---|")
    from mbe.mbe_pasn import bit_law
    for order in (1, 2):
        for lt in (True, False):
            a, c = bit_law(order, lt)
            print(f"| {order} | {'학습' if lt else '고정'} | "
                  f"{rule_budget(1.0, 1e-5, a=a, c=c)} |")

    print(f"\n**감사 결과: {'통과' if ok else '불일치 있음'}**")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
