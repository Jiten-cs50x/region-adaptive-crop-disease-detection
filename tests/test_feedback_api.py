from __future__ import annotations
import csv,hashlib,io,uuid
from datetime import datetime,timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from crop_adaptive.feedback import api

@pytest.fixture()
def client(tmp_path,monkeypatch):
 monkeypatch.setattr(api,"ROOT",tmp_path)
 monkeypatch.setattr(api,"UPLOADS",tmp_path/"uploads")
 monkeypatch.setattr(api,"DB_PATH",tmp_path/"feedback.sqlite3")
 api.UPLOADS.mkdir(parents=True)
 monkeypatch.setenv("CROP_SESSION_SECRET","test-session-secret-that-is-long-enough-123456")
 monkeypatch.setenv("AGRIVISION_ADMIN_USERNAME","admin-test")
 monkeypatch.setenv("AGRIVISION_ADMIN_PASSWORD","admin-password-123")
 config=tmp_path/"taxonomy.yaml"
 config.write_text("taxonomy:\n  classes:\n    - Healthy\n    - Brown Spot\n",encoding="utf-8")
 monkeypatch.setenv("CROP_CONFIG",str(config))
 with TestClient(api.app) as c:
  # Self-registration is available only for farmer accounts.
  for username in ("farmer-1","farmer-2"):
   r=c.post("/api/auth/register",json={"username":username,"display_name":username,"password":"farmer-password-123"})
   assert r.status_code==200
  # Only the administrator can create expert accounts.
  login(c,"admin-test","admin-password-123")
  for username in ("expert-9","expert-2"):
   r=c.post("/api/admin/experts",headers=csrf(c),json={"username":username,"display_name":username,"password":"expert-password-123"})
   assert r.status_code==200
  yield c

def login(client,username,password):
 r=client.post("/api/auth/login",json={"username":username,"password":password})
 assert r.status_code==200,r.text
 return {"X-CSRF-Token":r.headers["x-csrf-token"]}
def csrf(client):
 # Obtain a fresh admin session for setup requests.
 return login(client,"admin-test","admin-password-123")
def headers(client,role="farmer",user="farmer-1"):
 password="farmer-password-123" if role=="farmer" else "expert-password-123"
 return login(client,user,password)

def seed_prediction():
 image_path=api.UPLOADS/f"{uuid.uuid4()}.jpg"
 Image.new("RGB",(32,32),"green").save(image_path)
 digest=hashlib.sha256(image_path.read_bytes()).hexdigest()
 pid=str(uuid.uuid4())
 with api.db() as con:
  con.execute("INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?)",
   (pid,"farmer-1",str(image_path),digest,"Healthy",0.91,"test-model",datetime.now(timezone.utc).isoformat()))
 return pid

def test_registration_login_and_role_protection(client):
 assert client.get("/health").status_code==200
 assert client.get("/api/my-submissions").status_code==401
 farmer_headers=headers(client)
 assert client.get("/api/my-submissions",headers=farmer_headers).status_code==200
 expert_headers=headers(client,"expert","expert-9")
 assert client.get("/api/expert/queue",headers=expert_headers).status_code==200
 assert client.get("/api/expert/queue",headers=farmer_headers).status_code==403
 assert client.post("/api/admin/experts",headers=farmer_headers,json={"username":"intruder","display_name":"Intruder","password":"long-password-123"}).status_code==403

def test_full_farmer_expert_approval_workflow(client):
 pid=seed_prediction()
 submitted=client.post(f"/api/predictions/{pid}/feedback",headers=headers(client),json={"suggested_label":"Brown Spot","notes":"Visible brown lesions"})
 assert submitted.status_code==200
 fid=submitted.json()["feedback_id"]
 eh=headers(client,"expert","expert-9")
 queue=client.get("/api/expert/queue",headers=eh)
 assert queue.status_code==200 and queue.json()[0]["feedback_id"]==fid
 image=client.get(f"/api/expert/image/{pid}",headers=eh)
 assert image.status_code==200 and image.headers["content-type"]=="image/jpeg"
 approved=client.post(f"/api/expert/review/{fid}",headers=eh,json={"decision":"approve","verified_label":"Brown Spot","notes":"Expert confirmed"})
 assert approved.status_code==200 and approved.json()["status"]=="approved"
 mine=client.get("/api/my-submissions",headers=headers(client))
 assert mine.json()[0]["verified_label"]=="Brown Spot"
 assert mine.json()[0]["expert_notes"]=="Expert confirmed"
 history=client.get("/api/expert/reviews",headers=headers(client,"expert","expert-9"))
 assert history.status_code==200 and history.json()[0]["verified_label"]=="Brown Spot"
 exported=client.get("/api/expert/export.csv",headers=headers(client,"expert","expert-9"))
 rows=list(csv.DictReader(io.StringIO(exported.text)))
 assert len(rows)==1 and rows[0]["disease_label"]=="Brown Spot"
 assert "image_path" not in rows[0] and rows[0]["expert_notes"]=="Expert confirmed"

def test_ownership_self_review_and_invalid_label(client):
 pid=seed_prediction()
 assert client.post(f"/api/predictions/{pid}/feedback",headers=headers(client,"farmer","farmer-2"),json={"suggested_label":"Brown Spot"}).status_code==403
 assert client.post(f"/api/predictions/{pid}/feedback",headers=headers(client),json={"suggested_label":"Unknown"}).status_code==422
 sub=client.post(f"/api/predictions/{pid}/feedback",headers=headers(client),json={"suggested_label":"Brown Spot"})
 fid=sub.json()["feedback_id"]
 # Farmer identity cannot act as an expert even if it knows the endpoint.
 assert client.post(f"/api/expert/review/{fid}",headers=headers(client,"farmer","farmer-1"),json={"decision":"approve"}).status_code==403

def test_rejected_feedback_never_enters_verified_export(client):
 pid=seed_prediction()
 sub=client.post(f"/api/predictions/{pid}/feedback",headers=headers(client),json={"suggested_label":"Brown Spot"})
 rejected=client.post(f"/api/expert/review/{sub.json()['feedback_id']}",headers=headers(client,"expert","expert-2"),json={"decision":"reject","notes":"Insufficient evidence"})
 assert rejected.status_code==200 and rejected.json()["status"]=="rejected"
 export=client.get("/api/expert/export.csv",headers=headers(client,"expert","expert-2"))
 assert list(csv.DictReader(io.StringIO(export.text)))==[]

def test_csrf_and_logout_invalidate_session(client):
 h=headers(client)
 pid=seed_prediction()
 no_csrf=client.post(f"/api/predictions/{pid}/feedback",json={"suggested_label":"Brown Spot"})
 assert no_csrf.status_code==403
 assert client.post("/api/auth/logout",headers=h).status_code==200
 assert client.get("/api/my-submissions").status_code==401

def test_prediction_upload_runs_inference_and_rejects_duplicate(client,monkeypatch):
 import torch
 class StubModel:
  def __call__(self,batch): return torch.tensor([[0.1,0.9]])
 monkeypatch.setattr(api,"predictor",lambda:(StubModel(),["Healthy","Brown Spot"],lambda image:torch.zeros((3,224,224)),"test-checkpoint",torch))
 image_bytes=io.BytesIO();Image.new("RGB",(80,80),"green").save(image_bytes,format="PNG");payload=image_bytes.getvalue()
 h=headers(client)
 first=client.post("/api/predictions",headers=h,files={"image":("leaf.png",payload,"image/png")})
 assert first.status_code==200 and first.json()["predicted_label"]=="Brown Spot"
 duplicate=client.post("/api/predictions",headers=h,files={"image":("leaf.png",payload,"image/png")})
 assert duplicate.status_code==409
