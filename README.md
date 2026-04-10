# LunarLander RL Starter

Gymnasium `LunarLander-v3`를 Stable-Baselines3 `PPO`로 학습하고, 콘솔 평가와 GUI 시연까지 확인하는 최소 실행 가능 프로젝트입니다.

## 선택한 알고리즘: PPO

이 프로젝트는 `PPO`를 기본 알고리즘으로 사용합니다. `LunarLander-v3`는 이산 행동 환경이라 `DQN`도 사용할 수 있지만, 입문용으로는 Stable-Baselines3의 `PPO`가 설정할 값이 적고 학습이 비교적 안정적으로 시작되는 편입니다. 처음 목표가 "복잡한 튜닝보다 확실히 실행되는 구조"이므로 PPO를 선택했습니다.

## 파일 구조

```text
.
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── play_gui.py
├── train_visual.py
├── setup_venv.bat
├── run_train_100k.bat
├── run_train_300k.bat
├── run_train_visual.bat
├── run_evaluate.bat
├── run_play_gui.bat
└── run_play_gui_300k.bat
```

`train.py`로 학습을 저장하면 `models/ppo_lunarlander.zip` 파일이 생성됩니다. `train.py` 실행 중 TensorBoard 로그는 `runs/` 폴더에 생성될 수 있습니다. `train_visual.py`는 모델을 저장하지 않습니다.

## 설치

Python 3.10 이상을 권장합니다.

Windows 탐색기에서 바로 실행하려면 `setup_venv.bat`을 더블클릭하세요. `.venv` 가상환경을 만들고 `requirements.txt`를 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`LunarLander-v3`는 Box2D가 필요합니다. Windows에서 Box2D 또는 `box2d-py` 설치가 실패하면 아래 순서로 다시 시도하세요.

```powershell
pip install swig
pip install -r requirements.txt
```

그래도 실패하면 Visual Studio Build Tools의 C++ 빌드 도구가 필요한 환경일 수 있습니다.

## 빠른 테스트

모델 저장 없이 학습 루프가 돌아가는지만 확인합니다.

```powershell
python train.py --timesteps 100000 --no-save
```

학습 중간 과정을 GUI로 확인하려면 아래 파일을 실행합니다. 모델은 저장하지 않고, 기본값으로 처음 10개 학습 episode가 끝날 때마다 현재 정책을 GUI로 1 episode 시연합니다.

```powershell
python train_visual.py --timesteps 100000 --render-first-episodes 10 --render-every-episodes 0
```

Windows 탐색기에서는 `run_train_visual.bat`을 더블클릭하면 같은 설정으로 실행합니다.

모델까지 저장하려면 `--no-save`를 빼면 됩니다.

```powershell
python train.py --timesteps 100000
```

Windows 탐색기에서는 `run_train_100k.bat`을 더블클릭하면 `.venv`의 Python으로 100000 timesteps 학습을 실행하고 `models/ppo_lunarlander.zip`을 저장합니다.

학습 중에는 Stable-Baselines3 로그와 `[train] timesteps=...` 진행 메시지가 콘솔에 출력됩니다.

## 정식 학습

조금 더 나은 결과를 보려면 timesteps를 늘려 실험하세요.

```powershell
python train.py --timesteps 300000
```

Windows 탐색기에서는 `run_train_300k.bat`을 더블클릭하면 300000 timesteps 학습을 실행하고 `models/ppo_lunarlander_300k.zip`으로 저장합니다.

더 길게 학습하려면 다음처럼 실행합니다.

```powershell
python train.py --timesteps 1000000
```

기본 저장 위치는 `models/ppo_lunarlander.zip`입니다. 다른 이름으로 저장하려면:

```powershell
python train.py --timesteps 300000 --model-path models/ppo_lunarlander_300k
```

## 평가

저장된 모델을 GUI 없이 10 episode 평가하고 평균 보상을 출력합니다.

```powershell
python evaluate.py
```

Windows 탐색기에서는 `run_evaluate.bat`을 더블클릭하면 기본 모델 `models/ppo_lunarlander.zip`을 10 episode 평가합니다.

출력에는 각 episode reward, 10 episode 평균 보상, 표준편차, 성공 착륙 추정 횟수, 추락 추정 횟수가 포함됩니다. 성공/추락은 Gymnasium 정보값이 아니라 보상 기준 휴리스틱입니다. 기본값은 성공 `reward >= 200`, 추락 `reward <= -100`입니다.

다른 모델을 평가하려면:

```powershell
python evaluate.py --model-path models/ppo_lunarlander_300k.zip --episodes 10
```

## GUI 시연

Gymnasium의 human render 모드로 pygame 창을 열어 모델 행동을 직접 봅니다.

```powershell
python play_gui.py
```

Windows 탐색기에서는 `run_play_gui.bat`을 더블클릭하면 기본 모델 `models/ppo_lunarlander.zip`을 GUI로 3 episode 시연합니다.

300000 timesteps로 학습한 모델을 보려면 `run_play_gui_300k.bat`을 더블클릭하세요. 이 파일은 `models/ppo_lunarlander_300k.zip`을 GUI로 시연합니다.

다른 모델이나 episode 수를 지정하려면:

```powershell
python play_gui.py --model-path models/ppo_lunarlander_300k.zip --episodes 5
```

렌더링이 너무 빠르면 `--sleep`을 줄 수 있습니다.

```powershell
python play_gui.py --sleep 0.01
```

## 예상 소요 및 주의사항

- `100000` timesteps: 빠른 동작 확인용입니다. PC에 따라 수 분 정도 걸릴 수 있고, 아직 착륙 성능이 불안정할 수 있습니다.
- `300000` timesteps: 입문 실험용으로 더 적절합니다. 보상이 눈에 띄게 좋아질 가능성이 커집니다.
- `1000000` timesteps: 더 긴 학습입니다. 시간이 더 오래 걸리지만 성공 착륙 확률을 높이는 데 유리합니다.
- GUI 시연은 저장된 모델이 필요합니다. `--no-save`로만 학습했다면 먼저 `python train.py --timesteps 100000`처럼 저장 학습을 한 번 실행하세요.
- Box2D 설치 문제는 `gymnasium[box2d]`, `swig`, C++ 빌드 도구와 관련되는 경우가 많습니다.
