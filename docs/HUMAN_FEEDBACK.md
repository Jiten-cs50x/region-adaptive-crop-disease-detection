# Supervised human feedback (FastAPI)

This is an opt-in review service alongside the existing classifier. It does not alter the baseline checkpoint or automatically retrain/deploy a model.

## Install and run (Windows PowerShell)

```powershell
pip install -e ".[feedback]"
$env:FARMER_API_TOKEN="replace-with-a-long-random-farmer-token"
$env:EXPERT_API_TOKEN="replace-with-a-different-long-random-expert-token"
$env:CROP_MODEL_CHECKPOINT="runs/sugarcane_efficientnet_b0/best_model.pt"
$env:CROP_CONFIG="configs/sugarcane_karnataka.yaml"
uvicorn crop_adaptive.feedback.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Use distinct farmer and expert tokens and distinct user IDs. Never expose this development service directly to the public internet: deploy behind HTTPS, proper account authentication/authorization, rate limiting, secure secret storage, and managed image storage.

## Workflow

1. Farmer uploads a JPEG/PNG/WebP image (10 MB maximum). The configured checkpoint performs inference; client-supplied predictions are never accepted.
2. The farmer may submit one proposed correction for that prediction.
3. A separately authenticated expert reviews the image and proposal, then approves/rejects it. The submitter cannot review their own correction.
4. Only approved labels appear in `GET /api/expert/export.csv`. The export includes image paths, SHA-256 hashes, reviewer and verification timestamp. It is a review-data export, not a ready-to-train merged dataset.

## API

- `GET /health`
- `POST /api/predictions` multipart field `image` (farmer)
- `POST /api/predictions/{id}/feedback` JSON `{"suggested_label":"Healthy","notes":"..." }` (farmer)
- `GET /api/my-submissions` (farmer)
- `GET /api/expert/queue` (expert)
- `GET /api/expert/image/{prediction_id}` (expert)
- `POST /api/expert/review/{feedback_id}` JSON `{"decision":"approve","verified_label":"Healthy","notes":"..." }` (expert)
- `GET /api/expert/export.csv` (expert)

All protected endpoints require `Authorization: Bearer <role-token>` and `X-User-ID`. Labels are checked against the configured taxonomy. Original predictions are retained; review decisions and actions are append-recorded in SQLite audit/review tables. Duplicate image content is rejected by SHA-256.

## Retraining safety

Do not feed the CSV directly into the existing training command: its image paths are absolute and it is not a merged, leakage-checked training split. First stage approved images into a new dataset version, verify label quality and duplicates against train/validation/test, keep the original held-out test split untouched, create a new group-aware split, train a separate candidate checkpoint, evaluate it, and promote it only after explicit human approval. No deployment or checkpoint replacement is performed by this service.

The included token authentication is a minimal single-host prototype, not a substitute for production identity management. SQLite and local image storage are suitable for a demo/single process, not multi-worker production.

## Local verification

From the repository root in PowerShell, update the branch and install the test/runtime dependencies. If PyTorch and torchvision are already installed and working for the training environment, keep those versions rather than replacing them:

```powershell
git fetch origin
git switch feature/supervised-human-feedback
git pull
pip install -e ".[dev]"
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
pytest -q tests/test_feedback_api.py
```

Then configure the existing best checkpoint and start the API:

```powershell
$env:FARMER_API_TOKEN = "replace-with-a-long-random-farmer-token"
$env:EXPERT_API_TOKEN = "use-a-different-long-random-expert-token"
$env:CROP_MODEL_CHECKPOINT = "runs/sugarcane_efficientnet_b0/best_model.pt"
$env:CROP_CONFIG = "configs/sugarcane_karnataka.yaml"
uvicorn crop_adaptive.feedback.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The prediction path now uses the same configured deterministic prediction transform, class order, and model factory as training/evaluation; it fails closed if checkpoint classes do not exactly match the configured taxonomy. The integration tests cover authentication, ownership, correction submission, expert queue/image access, approval, rejection, and approved-only export. They seed test prediction rows and therefore do not validate the actual local checkpoint inference; perform one real image upload in the browser after the API tests pass.

Keep the server running while testing. Use a farmer ID such as `farmer-001` with the farmer token, then switch to a different expert ID such as `expert-001` with the expert token. Upload a fresh image (the same raw file hash is intentionally rejected if already submitted). Submit a correction, open the Expert tab, load the queue, inspect the image, and approve/reject. Refresh the farmer submissions and download the verified CSV as the expert. Only the approved item should appear in the CSV.

If tests fail, copy the complete traceback and the command that produced it. If inference returns 500, copy the server terminal traceback (remove tokens first). Do not share API tokens or checkpoint files.
