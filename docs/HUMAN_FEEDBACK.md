# Agrivision AI human feedback and accounts

The FastAPI service provides farmer image submissions, model predictions, correction proposals, expert verification, an audit trail and approved-feedback CSV export. It uses SQLite and local image storage by default. It does not automatically retrain or deploy a model.

## Account and role model

- Farmers can register themselves with a username, display name and password (minimum 10 characters). A phone number is optional.
- Experts cannot self-register. An administrator creates expert accounts from the Administrator workspace.
- The first administrator is bootstrapped from `AGRIVISION_ADMIN_USERNAME` and `AGRIVISION_ADMIN_PASSWORD` when the service starts and no account with that username exists.
- The selected dashboard workspace is not authority: the backend checks the signed-in account's stored role on every protected request.
- Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes, never plaintext.
- Successful login creates a signed, HttpOnly, SameSite=Lax session cookie with a 12-hour lifetime. Mutating requests also require a CSRF token held in page memory.
- Set `CROP_COOKIE_SECURE=1` behind HTTPS. The local default is `0` for localhost development only.
- Set a stable `CROP_SESSION_SECRET` of at least 32 characters. Do not use the sample values below on a public deployment.

## Install and run (Windows PowerShell)

Install the feedback dependencies:

```powershell
pip install -e ".[feedback]"
```

Configure the session secret and initial administrator. Generate the session secret with the first command; choose a strong, unique administrator password:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
$env:CROP_SESSION_SECRET = "PASTE_THE_GENERATED_VALUE_HERE"
$env:AGRIVISION_ADMIN_USERNAME = "admin"
$env:AGRIVISION_ADMIN_PASSWORD = "CHOOSE_A_LONG_UNIQUE_PASSWORD"
$env:CROP_COOKIE_SECURE = "0"
$env:CROP_MODEL_CHECKPOINT = "runs/sugarcane_efficientnet_b0/best_model.pt"
$env:CROP_CONFIG = "configs/sugarcane_karnataka.yaml"
uvicorn crop_adaptive.feedback.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Sign in as the initial administrator, create approved expert accounts, and share their credentials securely. Farmers can use the Farmer workspace to register. Experts use credentials supplied by the administrator.

If the administrator username already exists, changing the bootstrap environment variables does not reset that account's password. Account recovery/reset is not yet implemented.

## Local data storage

By default, the service stores data under `runs/feedback/`:

- `feedback.sqlite3`: user accounts, prediction metadata, feedback, expert reviews and audit records.
- `uploads/`: uploaded images stored as JPEG files.

Set `CROP_FEEDBACK_HOME` to move both to another persistent directory. Back up the entire directory while the service is stopped (or use a consistent SQLite backup procedure). Do not commit this directory to Git. SQLite and local files are suitable for a local demonstration or single-host deployment, not a horizontally scaled production service.

## Workflow

1. A farmer registers or signs in, then uploads a JPEG/PNG/WebP image (10 MB maximum).
2. The configured checkpoint performs inference; client-supplied predictions are never accepted.
3. The farmer may submit one proposed correction for that prediction.
4. An independently authenticated expert reviews the image and proposal, then approves/rejects it. The submitter cannot verify their own correction.
5. Only approved labels appear in `GET /api/expert/export.csv`. The export contains prediction IDs, image SHA-256 hashes, verified labels, reviewer IDs, notes and timestamps, not server filesystem paths.

## API

- `POST /api/auth/register` — farmer self-registration only
- `POST /api/auth/login` — username/password sign-in
- `GET /api/auth/me` — current account
- `POST /api/auth/logout` — invalidate the current account's sessions
- `POST /api/admin/experts` — administrator creates an expert
- `GET /api/admin/experts` — administrator lists expert accounts
- `GET /health`
- `GET /api/taxonomy`
- `POST /api/predictions` — farmer image upload/inference
- `POST /api/predictions/{id}/feedback` — farmer correction
- `GET /api/my-submissions` — farmer history
- `GET /api/expert/queue` — expert review queue
- `GET /api/expert/image/{prediction_id}` — expert image
- `POST /api/expert/review/{feedback_id}` — expert decision
- `GET /api/expert/export.csv` — approved-only export
- `GET /api/expert/reviews` — expert review history

Browser requests use the session cookie automatically. The dashboard attaches the CSRF token returned in the `X-CSRF-Token` response header to state-changing requests. Do not reintroduce user-editable identity headers or shared bearer tokens.

## Retraining safety

Do not feed the CSV directly into the existing training command. First stage approved images into a new dataset version, verify label quality and duplicates against train/validation/test, keep the original held-out test split untouched, create a new group-aware split, train a separate candidate checkpoint, evaluate it, and promote it only after explicit human approval. No deployment or checkpoint replacement is performed by this service.

## Security and deployment limitations

This is an improved single-host project authentication flow, not a production security audit. Before public deployment, add HTTPS, rate limiting, account recovery, email/phone verification as appropriate, administrator password reset and expert deactivation UI, secure secret management, persistent managed image storage, database backups, monitoring and a production database such as PostgreSQL. Configure `CROP_COOKIE_SECURE=1` behind HTTPS. Do not expose the development server directly to the public internet.

## Tests

```powershell
pip install -e ".[dev]"
pytest -q tests/test_feedback_api.py
```

Tests cover farmer registration, admin-created experts, role authorization, CSRF/session invalidation, ownership, feedback submission, expert approval/rejection, review history, approved-only export and mocked prediction upload. They do not replace a real browser walkthrough with the local model checkpoint.
