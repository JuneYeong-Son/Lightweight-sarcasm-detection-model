# Android에서 성능 측정하기

PyTorch 원본 모델(`model/`)과 ONNX 모델(`onnx/`)을 Android 폰에서 같은 조건으로 비교하는 방법이다.
측정 스크립트는 [scripts/benchmark.py](../scripts/benchmark.py)이고, PC와 폰에서 같은 명령으로 실행한다.

## 방식

Termux 앱 안에 Ubuntu(proot-distro)를 띄우고, PC와 같은 Python 패키지로 실행한다.

- 장점: 코드 수정 없이 PyTorch와 ONNX를 같은 환경에서 비교할 수 있다.
- 한계: 실제 Android 앱(ONNX Runtime Android, ExecuTorch)보다 약간 느릴 수 있다. proot 오버헤드는 주로 파일·시스템 호출에서 생기고 행렬 연산에는 영향이 적다. 두 모델이 같은 환경에서 돌기 때문에 **상대 비교**에는 충분하다.

## 준비물

- Android 폰 (ARM64, 64비트)
- 저장공간 약 4GB 이상 (모델 약 0.9GB + 패키지 약 2GB)
- RAM 4GB 이상 권장 (측정 중 프로세스 최대 메모리 약 1.1GB)
- Wi-Fi (패키지, 데이터셋 다운로드)

## 1. Termux 설치

Play 스토어 버전은 업데이트가 멈춰 있으므로 **F-Droid** 또는 **Termux GitHub Releases**에서 받는다.

## 2. Ubuntu 설치 (Termux에서)

```bash
pkg update && pkg upgrade -y
pkg install -y proot-distro
proot-distro install ubuntu
proot-distro login ubuntu
```

이후 단계는 모두 Ubuntu 안에서 실행한다. 다음에 다시 들어갈 때는 `proot-distro login ubuntu`만 실행하면 된다.

## 3. Python 환경 (Ubuntu 안에서)

```bash
apt update
apt install -y python3 python3-venv git git-lfs

python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip

# PyTorch는 CPU 전용 버전을 먼저 설치한다 (기본 버전은 CUDA 패키지까지 받아서 매우 큼)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. 코드와 모델 받기

```bash
git lfs install
git clone -b feature/origin-onnx https://github.com/JuneYeong-Son/Lightweight-sarcasm-detection-model.git
cd Lightweight-sarcasm-detection-model
git lfs pull                      # model.safetensors, model.onnx 실제 파일 받기
pip install -r requirements-benchmark.txt
```

- 저장소가 비공개라면 비밀번호 대신 GitHub Personal Access Token을 입력해야 한다.
- `ls -lh model onnx`로 확인했을 때 모델 파일이 약 420MB가 아니라 몇백 바이트라면 `git lfs pull`이 안 된 것이다.

## 5. 측정 실행

```bash
source ~/venv/bin/activate
cd ~/Lightweight-sarcasm-detection-model

# (1) 먼저 속도만 빠르게 확인 (수 분)
python scripts/benchmark.py --no-eval --device-name 기종이름

# (2) 정확도/F1까지 전체 측정 (validation 1321개, 수십 분 걸릴 수 있음)
python scripts/benchmark.py --device-name 기종이름
```

`기종이름`에는 `galaxy-s24`처럼 영문으로 기기를 구분할 이름을 넣는다. 결과 파일 이름에 쓰인다.

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--threads` | 4 | 추론 스레드 수. 폰은 큰 코어 수에 맞출 때 보통 가장 빠르다. 4와 8을 둘 다 재 보는 것을 권장 |
| `--no-eval` | 꺼짐 | 정확도/F1 평가를 건너뛰고 속도·메모리만 측정 |
| `--eval-limit` | 전체 | 정확도 평가에 쓸 데이터 개수. 시간이 너무 걸리면 `--eval-limit 300` 등으로 줄인다 |
| `--split` | validation | `test`로 바꾸면 test 셋(1037개)으로 평가 |
| `--backends` | pytorch onnx | 하나만 재려면 `--backends onnx` |
| `--runs` / `--warmup` | 50 / 5 | 지연시간 측정 횟수 / 측정 전 예열 횟수 |
| `--cooldown` | 10 | 두 모델 측정 사이 대기 시간(초). 발열 영향 완화 |

## 6. 정확하게 재기 위한 팁

- 다른 앱을 모두 닫고, 절전 모드를 끈다.
- Termux 알림에서 **Acquire wakelock**을 눌러 화면이 꺼져도 측정이 멈추지 않게 한다.
- 폰이 뜨거우면 성능이 떨어진다(스로틀링). 측정 사이에 식히고, 같은 측정을 2~3번 반복해 결과가 비슷한지 본다.
- 충전 중에는 발열이 커지므로, 가능하면 배터리 50% 이상에서 충전기를 빼고 잰다.

## 7. 결과 확인

- 화면에 PyTorch와 ONNX를 나란히 비교한 표가 출력된다.
- 상세 결과는 `results/benchmark/날짜_기종이름.json`에 저장된다. 기기 정보, 설정, 라이브러리 버전이 함께 기록된다.
- PC로 가져오려면 `git add results/benchmark && git commit && git push`를 하거나, 파일을 직접 복사한다.

### 측정 항목

| 항목 | 의미 |
|---|---|
| 파일 크기 | 모델 파일 크기 |
| 모델 로드 | 모델 파일을 읽어 추론 준비까지 걸린 시간 |
| 지연시간 | 문장쌍 1개(batch 1) 추론 시간. 토크나이저 시간은 제외. validation 앞 32개 데이터를 돌아가며 사용 |
| 메모리 | 모델 로드로 늘어난 메모리와 프로세스 최대 메모리 |
| 정확도 / F1 | 비꼼(Sarcasm)을 양성 클래스로 한 이진 F1 |
| 예측 일치율 | 두 모델의 예측이 같은 비율. ONNX 변환이 정확하면 1.0에 가깝다 |
