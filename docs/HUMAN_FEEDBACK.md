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
