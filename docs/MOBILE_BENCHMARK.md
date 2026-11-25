# 모바일에서 성능 측정하기

ONNX 모델(① ~ ⑤)을 Android 폰에서 측정하고 PC 결과와 비교한다.
스크립트는 [pipeline/scripts/mobile_benchmark.py](../pipeline/scripts/mobile_benchmark.py)이고 PC와 폰에서 같은 명령으로 실행한다.
`onnxruntime`, `tokenizers`, `numpy`만 쓰므로 폰에 PyTorch, transformers, datasets를 설치하지 않아도 된다.

## 무엇을 재나

| 항목 | 모바일에서 보는 이유 |
|---|---|
| 모델 로드, 첫 추론 (ms) | 앱을 켜고 처음 판단할 때까지의 체감 시간. 저장장치 속도에 따라 달라진다 |
| 추론 / 토크나이저 / 전체 지연시간 | batch 1. 실제 앱에서 걸리는 시간은 토크나이저까지 포함한 전체 지연시간이다 |
| 지속 부하 (`--sustained-seconds`) | 계속 돌리면 발열 때문에 클럭이 내려가 느려지는지 본다. 일정 간격마다 지연시간, 온도, 코어별 클럭을 기록한다 |
| 메모리 | 로드로 늘어난 RSS와 프로세스 최대 RSS. 폰은 RAM이 작아 PC보다 중요하다 |
| 정확도 / F1 / BA / Macro-F1 | 앱과 같은 조건(문장마다 batch 1, 패딩 없음)으로 평가한다 |
| PC 대비 예측 일치율, logit 차이 (`--reference`) | ARM과 x86은 INT8 커널이 달라 같은 모델이라도 출력이 조금 달라질 수 있다. 그 차이를 직접 확인한다 |

PC에서는 스레드 수(1, 4)만 바꾸면 logit이 완전히 같게 나온다. 그래서 폰에서 차이가 나면 기기·런타임 차이라고 보면 된다.

## 1. PC에서 준비

```powershell
# (1) 평가 데이터를 파일로 저장 -> data/kocosa_validation.jsonl, data/kocosa_test.jsonl (datasets 필요, 한 번만)
python pipeline/scripts/mobile_benchmark.py prepare

# (2) PC 기준 결과 만들기 -> results/mobile/<시각>_pc_test.json, .npz
python pipeline/scripts/mobile_benchmark.py run --device-name pc
```

`data/` 폴더와 PC 결과 `.npz`는 폰으로 옮겨야 한다. 커밋해서 `git pull`로 받거나, USB나 `adb push`로 복사한다.

## 2. 폰 환경 (Termux + Ubuntu)

Termux는 **F-Droid**나 **GitHub Releases**에서 받는다(Play 스토어 버전은 업데이트가 멈춰 있다).

```bash
# Termux에서
pkg update && pkg upgrade -y
pkg install -y proot-distro
proot-distro install ubuntu
proot-distro login ubuntu

# Ubuntu 안에서
apt update && apt install -y python3 python3-venv git git-lfs
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install --upgrade pip

git lfs install
git clone -b feature/prunning_o_DynINT8 https://github.com/JuneYeong-Son/Lightweight-sarcasm-detection-model.git
cd Lightweight-sarcasm-detection-model
git lfs pull
pip install -r pipeline/requirements-mobile.txt
```

`ls -lh onnx-pruned-int8`로 확인했을 때 `model.onnx`가 수백 바이트라면 `git lfs pull`이 안 된 것이다.

## 3. 폰에서 측정

```bash
source ~/venv/bin/activate && cd ~/Lightweight-sarcasm-detection-model
REF=results/mobile/<PC 결과>.npz

# (1) 빠른 확인: 속도·메모리만 (몇 분)
python pipeline/scripts/mobile_benchmark.py run --device-name galaxy-s24 --no-eval

# (2) 본 측정: 정확도 + PC 비교 + 2분 지속 부하
python pipeline/scripts/mobile_benchmark.py run --device-name galaxy-s24 \
    --reference $REF --sustained-seconds 120

# (3) 스레드 수 비교 (속도만): 폰은 큰 코어 수에 맞출 때 보통 가장 빠르다
python pipeline/scripts/mobile_benchmark.py run --device-name galaxy-s24 --no-eval --threads 1 2 4 8
```

다른 변형(① ONNX fp32, ② Dynamic INT8, ④·⑤ QAT)도 `이름=폴더`로 넣어 함께 잴 수 있다. 같은 실행에서 함께 재야 공정하게 비교할 수 있다.

```bash
python pipeline/scripts/mobile_benchmark.py run --device-name galaxy-s24 \
    --models onnx=../other/onnx onnx-int8=../other/onnx-int8 pruned-int8=onnx-pruned-int8
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models` | `pruned-fp32`, `pruned-int8` | `이름=폴더` 목록. 폴더에 `model.onnx`와 `tokenizer.json`이 있어야 한다 |
| `--threads` | 4 | 여러 개를 주면 모두 측정한다 (예: `1 2 4 8`) |
| `--split` | test | `validation`도 가능하다 |
| `--no-eval` / `--eval-limit` | 꺼짐 / 전체 | 정확도 평가를 생략하거나 문장 수를 줄인다 |
| `--sustained-seconds` / `--sustained-window` | 0 / 10 | 지속 부하 측정 시간과 기록 간격(초) |
| `--reference` | 없음 | 비교 기준 `.npz`. 모델 이름이 같은 결과끼리 비교한다 |
| `--warmup` / `--runs` | 10 / 100 | 지연시간 측정 전 예열 횟수 / 측정 횟수 |
| `--cooldown` | 30 | 모델 사이 대기 시간(초). 앞 모델에서 생긴 열을 식힌다 |

## 4. 정확하게 재기 위한 팁

- 다른 앱을 모두 닫고 절전 모드를 끈다. 충전기는 빼고 배터리 50% 이상에서 잰다.
- Termux 알림에서 **Acquire wakelock**을 눌러 화면이 꺼져도 측정이 멈추지 않게 한다.
- 폰이 식은 상태에서 시작하고, 같은 측정을 2~3번 반복해 결과가 비슷한지 확인한다. 결과 JSON에 측정 전후 온도가 기록된다.
- 온도와 클럭은 Android 버전에 따라 읽기가 막혀 있을 수 있다. 그때는 `null`로 기록되고 지연시간 변화만으로 판단한다.
- proot 환경은 실제 앱(ONNX Runtime Android)보다 조금 느릴 수 있지만, 모델들이 같은 환경에서 돌기 때문에 **상대 비교**에는 충분하다.
- 배터리 소모(전력)는 루트 권한 없이는 정확히 잴 수 없어 측정하지 않는다.

## 5. 결과

화면에 모델별 비교표가 출력되고 `results/mobile/`에 두 파일이 저장된다.

| 파일 | 내용 |
|---|---|
| `<시각>_<기종>_<split>.json` | 기기 정보(기종, SoC, 코어별 최대 클럭, 배터리, 온도), 설정, 모델별 결과, 지속 부하 타임라인 |
| `<시각>_<기종>_<split>.npz` | 모델별 logits. 다른 기기에서 `--reference`로 쓴다 |
