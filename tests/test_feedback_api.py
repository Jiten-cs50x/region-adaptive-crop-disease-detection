from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from crop_adaptive.feedback import api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "feedback.sqlite3")
    api.UPLOADS.mkdir(parents=True)
    monkeypatch.setenv("FARMER_API_TOKEN", "farmer-test-token")
    monkeypatch.setenv("EXPERT_API_TOKEN", "expert-test-token")
    config = tmp_path / "taxonomy.yaml"
    config.write_text(
        "taxonomy:\n  classes:\n    - Healthy\n    - Brown Spot\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CROP_CONFIG", str(config))
    with api.db() as con:
        con.executescript("""
        CREATE TABLE predictions(id TEXT PRIMARY KEY,farmer_id TEXT NOT NULL,image_path TEXT NOT NULL,image_sha256 TEXT NOT NULL UNIQUE,predicted_label TEXT NOT NULL,confidence REAL NOT NULL,model_version TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE feedback(id TEXT PRIMARY KEY,prediction_id TEXT NOT NULL REFERENCES predictions(id),farmer_id TEXT NOT NULL,suggested_label TEXT NOT NULL,notes TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(prediction_id,farmer_id));
        CREATE TABLE expert_reviews(id TEXT PRIMARY KEY,feedback_id TEXT NOT NULL REFERENCES feedback(id),expert_id TEXT NOT NULL,decision TEXT NOT NULL,verified_label TEXT,notes TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT,actor_id TEXT NOT NULL,action TEXT NOT NULL,object_id TEXT NOT NULL,details TEXT NOT NULL,created_at TEXT NOT NULL);
        """)
    with TestClient(api.app) as test_client:
        yield test_client


def seed_prediction(farmer="farmer-1"):
    image_path = api.UPLOADS / f"{uuid.uuid4()}.jpg"
    Image.new("RGB", (32, 32), "green").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    prediction_id = str(uuid.uuid4())
    with api.db() as con:
        con.execute(
            "INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?)",
            (prediction_id, farmer, str(image_path), digest, "Healthy", 0.91, "test-model", datetime.now(timezone.utc).isoformat()),
        )
    return prediction_id


def headers(role="farmer", user="farmer-1"):
    token = "farmer-test-token" if role == "farmer" else "expert-test-token"
    return {"Authorization": f"Bearer {token}", "X-User-ID": user}


def test_health_and_authentication(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/my-submissions", headers={"X-User-ID": "farmer-1"}).status_code == 401
    assert client.get("/api/my-submissions", headers=headers()).status_code == 200


def test_full_farmer_expert_approval_workflow(client):
    prediction_id = seed_prediction()
    submitted = client.post(
        f"/api/predictions/{prediction_id}/feedback",
        headers=headers(),
        json={"suggested_label": "Brown Spot", "notes": "Visible brown lesions"},
    )
    assert submitted.status_code == 200
    feedback_id = submitted.json()["feedback_id"]
    assert submitted.json()["status"] == "pending_expert_verification"

    queue = client.get("/api/expert/queue", headers=headers("expert", "expert-9"))
    assert queue.status_code == 200
    assert len(queue.json()) == 1
    assert queue.json()[0]["feedback_id"] == feedback_id

    image = client.get(f"/api/expert/image/{prediction_id}", headers=headers("expert", "expert-9"))
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"

    approved = client.post(
        f"/api/expert/review/{feedback_id}",
        headers=headers("expert", "expert-9"),
        json={"decision": "approve", "verified_label": "Brown Spot", "notes": "Expert confirmed"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    mine = client.get("/api/my-submissions", headers=headers())
    assert mine.status_code == 200
    assert mine.json()[0]["status"] == "approved"

    exported = client.get("/api/expert/export.csv", headers=headers("expert", "expert-9"))
    assert exported.status_code == 200
    rows = list(csv.DictReader(io.StringIO(exported.text)))
    assert len(rows) == 1
    assert rows[0]["disease_label"] == "Brown Spot"
    assert rows[0]["expert_id"] == "expert-9"


def test_rejects_wrong_owner_self_review_and_invalid_label(client):
    prediction_id = seed_prediction()
    wrong_owner = client.post(
        f"/api/predictions/{prediction_id}/feedback",
        headers=headers(user="farmer-2"),
        json={"suggested_label": "Brown Spot"},
    )
    assert wrong_owner.status_code == 403

    invalid_label = client.post(
        f"/api/predictions/{prediction_id}/feedback",
        headers=headers(),
        json={"suggested_label": "Made Up Disease"},
    )
    assert invalid_label.status_code == 422

    submitted = client.post(
        f"/api/predictions/{prediction_id}/feedback",
        headers=headers(),
        json={"suggested_label": "Brown Spot"},
    )
    feedback_id = submitted.json()["feedback_id"]
    self_review = client.post(
        f"/api/expert/review/{feedback_id}",
        headers=headers("expert", "farmer-1"),
        json={"decision": "approve"},
    )
    assert self_review.status_code == 403


def test_reject_decision_never_enters_verified_export(client):
    prediction_id = seed_prediction()
    submitted = client.post(
        f"/api/predictions/{prediction_id}/feedback",
        headers=headers(),
        json={"suggested_label": "Brown Spot"},
    )
    feedback_id = submitted.json()["feedback_id"]
    rejected = client.post(
        f"/api/expert/review/{feedback_id}",
        headers=headers("expert", "expert-2"),
        json={"decision": "reject", "notes": "Insufficient evidence"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    exported = client.get("/api/expert/export.csv", headers=headers("expert", "expert-2"))
    assert list(csv.DictReader(io.StringIO(exported.text))) == []
