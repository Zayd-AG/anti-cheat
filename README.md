# Game Anti-Cheat Detection System

A backend project that scores simulated player sessions and keeps a Cheat Risk Score for each player. It compares a simple z-score baseline with an Isolation Forest model, then serves scores through an asynchronous API. All data and results are synthetic.

## How it works

```text
Game client -> FastAPI POST /events -> Redis/Celery queue -> Celery scorer -> scaled Isolation Forest
                                                                    |              |
                                                                    |              -> Redis cheat_risk:{session_id}
                                                                    -> PostgreSQL session history + rolling player risk
Game client <- FastAPI GET /cheat_risk/{session_id} <- Redis cache
Game client <- FastAPI GET /players/{player_id}/cheat_risk <- PostgreSQL summary
```

FastAPI accepts player events without waiting for the model. Celery scores each session in the background, which keeps the API responsive. Redis queues jobs and caches recent session results, while PostgreSQL stores session history and a rolling score from each player's latest 20 sessions.

## Setup

This project targets Python 3.11. Docker is the simplest reproducible path:

```bash
docker compose build
docker compose run --rm api python generate_data.py
docker compose run --rm api python baseline_detector.py
docker compose run --rm api python ml_detector.py
docker compose up --build
```

Train the model before starting the worker. With the stack running in a second terminal:

```bash
docker compose run --rm api python load_test.py --url http://api:8000 --sessions 100 --rate 10
docker compose run --rm api python imbalance_experiment.py
```

For a non-Docker installation, create a Python 3.11 virtual environment and run `pip install -r requirements.txt` first.

## What each part does

1. `generate_data.py` creates 5,000 sessions across typical, elite, controller, and lag-affected legitimate players, plus four cheat styles.
2. `baseline_detector.py` uses z-scores as an explainable starting point and shows where simple rules make mistakes.
3. `ml_detector.py` trains Isolation Forest on a 60/20/20 train/validation/test split. The validation split selects a false-positive-rate threshold, and the untouched test split produces the final metrics.
4. `app.py` and `tasks.py` accept player events, score them in the background, cache each session in Redis, and save player history in PostgreSQL.
5. `load_test.py` measures end-to-end scoring latency. `imbalance_experiment.py` shows how model performance changes at different cheat rates.

## API example

```bash
curl -X POST http://localhost:8000/events -H "Content-Type: application/json" -d '{
  "session_id":"match-123-player-7",
  "player_id":"player-7",
  "reaction_times_ms":[225,245,271],
  "movement_speeds":[4.8,5.2,6.0],
  "click_timestamps_ms":[0,205,430,660],
  "aim_movements":[false,false,true,false]
}'
curl http://localhost:8000/cheat_risk/match-123-player-7
curl http://localhost:8000/players/player-7/cheat_risk
```

`POST /events` adds a session to the scoring queue and returns a task ID. `GET /cheat_risk/{session_id}` returns that session's score once it is ready. `GET /players/{player_id}/cheat_risk` returns a player's rolling score from their latest 20 sessions.

## Generated evaluation graphs

Run the commands above to create these repository-local PNGs:

| Graph | Script |
| --- | --- |
| `graphs/baseline_vs_ml.png` | `ml_detector.py` |
| `graphs/anomaly_score_distribution.png` | `ml_detector.py` |
| `graphs/precision_recall_vs_cheat_rate.png` | `imbalance_experiment.py` |
| `graphs/held_out_evaluation.json` | `ml_detector.py` |

### Baseline vs ML precision/recall

![Baseline vs Isolation Forest comparison](graphs/baseline_vs_ml.png)

### Isolation Forest anomaly-score distribution

![Anomaly-score distribution](graphs/anomaly_score_distribution.png)

### Precision and recall by cheat rate

![Precision and recall versus cheat rate](graphs/precision_recall_vs_cheat_rate.png)

## Limitations

The data and results are synthetic, so they do not represent real player behavior perfectly. A real system would need testing on production-like data, human review, and an appeal process; these scores should not trigger automatic punishment.
