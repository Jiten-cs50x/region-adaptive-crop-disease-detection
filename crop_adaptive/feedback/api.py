from __future__ import annotations
import csv, hashlib, io, os, secrets, sqlite3, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

ROOT=Path(os.getenv("CROP_FEEDBACK_HOME","runs/feedback")).resolve()
UPLOADS=ROOT/"uploads"; DB_PATH=ROOT/"feedback.sqlite3"
ROOT.mkdir(parents=True,exist_ok=True); UPLOADS.mkdir(parents=True,exist_ok=True)
app=FastAPI(title="Crop Disease Human Feedback API",version="1.0.0")
def now(): return datetime.now(timezone.utc).isoformat()
@contextmanager
def db():
 c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row; c.execute("PRAGMA foreign_keys=ON")
 try: yield c; c.commit()
 except Exception: c.rollback(); raise
 finally: c.close()
with db() as c:
 c.executescript("""
 CREATE TABLE IF NOT EXISTS predictions(id TEXT PRIMARY KEY,farmer_id TEXT NOT NULL,image_path TEXT NOT NULL,image_sha256 TEXT NOT NULL UNIQUE,predicted_label TEXT NOT NULL,confidence REAL NOT NULL,model_version TEXT NOT NULL,created_at TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY,prediction_id TEXT NOT NULL REFERENCES predictions(id),farmer_id TEXT NOT NULL,suggested_label TEXT NOT NULL,notes TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(prediction_id,farmer_id));
 CREATE TABLE IF NOT EXISTS expert_reviews(id TEXT PRIMARY KEY,feedback_id TEXT NOT NULL REFERENCES feedback(id),expert_id TEXT NOT NULL,decision TEXT NOT NULL,verified_label TEXT,notes TEXT NOT NULL,created_at TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT,actor_id TEXT NOT NULL,action TEXT NOT NULL,object_id TEXT NOT NULL,details TEXT NOT NULL,created_at TEXT NOT NULL);
 CREATE INDEX IF NOT EXISTS feedback_status_idx ON feedback(status);
 """)
def auth(role):
 async def check(authorization:Optional[str]=Header(default=None)):
  token=os.getenv(role.upper()+"_API_TOKEN")
  if not token: raise HTTPException(503,f"Configure {role.upper()}_API_TOKEN")
  supplied=authorization.removeprefix("Bearer ").strip() if authorization else ""
  if not secrets.compare_digest(supplied,token): raise HTTPException(401,"Invalid bearer token")
  return role
 return check
def classes():
 import yaml
 p=Path(os.getenv("CROP_CONFIG","configs/sugarcane_karnataka.yaml"))
 if not p.is_file(): raise HTTPException(503,"CROP_CONFIG taxonomy file not found")
 return yaml.safe_load(p.read_text(encoding="utf-8"))["taxonomy"]["classes"]
def predictor():
 p=Path(os.getenv("CROP_MODEL_CHECKPOINT",""))
 if not p.is_file(): raise HTTPException(503,"Set CROP_MODEL_CHECKPOINT to enable inference")
 try:
  import torch
  from torchvision import transforms
  from crop_adaptive.models.efficientnet import build_model
  ckpt=torch.load(p,map_location="cpu",weights_only=False); labels=ckpt.get("classes") or list(ckpt["class_to_idx"])
  model=build_model(len(labels),pretrained=False); model.load_state_dict(ckpt.get("model_state_dict",ckpt)); model.eval()
  tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
  return model,labels,tf,hashlib.sha256(p.read_bytes()).hexdigest()[:12],torch
 except Exception as e: raise HTTPException(500,f"Checkpoint load failed: {e}")
class Correction(BaseModel):
 suggested_label:str=Field(min_length=1,max_length=100); notes:str=Field(default="",max_length=2000)
class Review(BaseModel):
 decision:str=Field(pattern="^(approve|reject)$"); verified_label:Optional[str]=None; notes:str=Field(default="",max_length=2000)
@app.get("/health")
def health(): return {"status":"ok","database":str(DB_PATH)}
@app.get("/",response_class=HTMLResponse)
def home(): return HTMLResponse((Path(__file__).with_name("dashboard.html")).read_text(encoding="utf-8"))
@app.post("/api/predictions")
async def predict(image:UploadFile=File(...),farmer_id:str=Header(alias="X-User-ID"),_=Depends(auth("farmer"))):
 if image.content_type not in {"image/jpeg","image/png","image/webp"}: raise HTTPException(415,"Upload JPEG, PNG or WebP")
 raw=await image.read(10*1024*1024+1)
 if len(raw)>10*1024*1024: raise HTTPException(413,"Image exceeds 10 MB")
 try:
  with Image.open(io.BytesIO(raw)) as im: im.verify()
  with Image.open(io.BytesIO(raw)) as im: rgb=im.convert("RGB")
 except (UnidentifiedImageError,OSError): raise HTTPException(400,"Invalid image")
 model,labels,tf,version,torch=predictor()
 with torch.inference_mode():
  prob=torch.softmax(model(tf(rgb).unsqueeze(0)),dim=1)[0]; conf,idx=prob.max(0)
 label=labels[int(idx)]; digest=hashlib.sha256(raw).hexdigest(); pid=str(uuid.uuid4()); path=UPLOADS/(pid+".jpg")
 rgb.save(path,"JPEG",quality=92)
 try:
  with db() as c:
   c.execute("INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?)",(pid,farmer_id,str(path),digest,label,float(conf),version,now()))
   c.execute("INSERT INTO audit_log(actor_id,action,object_id,details,created_at) VALUES(?,?,?,?,?)",(farmer_id,"prediction_created",pid,label,now()))
 except sqlite3.IntegrityError:
  path.unlink(missing_ok=True); raise HTTPException(409,"This image has already been submitted")
 return {"id":pid,"predicted_label":label,"confidence":float(conf),"model_version":version}
@app.post("/api/predictions/{pid}/feedback")
def correct(pid:str,payload:Correction,farmer_id:str=Header(alias="X-User-ID"),_=Depends(auth("farmer"))):
 if payload.suggested_label not in classes(): raise HTTPException(422,"Label not in configured taxonomy")
 with db() as c:
  p=c.execute("SELECT * FROM predictions WHERE id=?",(pid,)).fetchone()
  if not p: raise HTTPException(404,"Prediction not found")
  if p["farmer_id"]!=farmer_id: raise HTTPException(403,"Only the submitting farmer can correct this prediction")
  if payload.suggested_label==p["predicted_label"]: raise HTTPException(400,"Correction must differ from model label")
  fid=str(uuid.uuid4())
  try: c.execute("INSERT INTO feedback VALUES(?,?,?,?,?,?,?)",(fid,pid,farmer_id,payload.suggested_label,payload.notes,"pending",now()))
  except sqlite3.IntegrityError: raise HTTPException(409,"Feedback already submitted")
  c.execute("INSERT INTO audit_log(actor_id,action,object_id,details,created_at) VALUES(?,?,?,?,?)",(farmer_id,"feedback_submitted",fid,payload.suggested_label,now()))
 return {"feedback_id":fid,"status":"pending_expert_verification"}
@app.get("/api/my-submissions")
def mine(farmer_id:str=Header(alias="X-User-ID"),_=Depends(auth("farmer"))):
 with db() as c: rows=c.execute("SELECT p.id,p.predicted_label,p.confidence,p.created_at,f.suggested_label,f.status FROM predictions p LEFT JOIN feedback f ON f.prediction_id=p.id WHERE p.farmer_id=? ORDER BY p.created_at DESC",(farmer_id,)).fetchall()
 return [dict(x) for x in rows]
@app.get("/api/expert/queue")
def queue(expert_id:str=Header(alias="X-User-ID"),_=Depends(auth("expert"))):
 with db() as c: rows=c.execute("SELECT f.id feedback_id,f.prediction_id,f.farmer_id,f.suggested_label,f.notes,p.predicted_label,p.confidence FROM feedback f JOIN predictions p ON p.id=f.prediction_id WHERE f.status='pending' ORDER BY f.created_at").fetchall()
 return [dict(x) for x in rows]
@app.get("/api/expert/image/{pid}")
def expert_image(pid:str,expert_id:str=Header(alias="X-User-ID"),_=Depends(auth("expert"))):
 with db() as c: p=c.execute("SELECT image_path FROM predictions WHERE id=?",(pid,)).fetchone()
 if not p or not Path(p["image_path"]).is_file(): raise HTTPException(404,"Image not found")
 return FileResponse(p["image_path"])
@app.post("/api/expert/review/{fid}")
def review(fid:str,payload:Review,expert_id:str=Header(alias="X-User-ID"),_=Depends(auth("expert"))):
 with db() as c:
  f=c.execute("SELECT f.*,p.farmer_id FROM feedback f JOIN predictions p ON p.id=f.prediction_id WHERE f.id=?",(fid,)).fetchone()
  if not f: raise HTTPException(404,"Feedback not found")
  if f["farmer_id"]==expert_id: raise HTTPException(403,"Submitter cannot verify their own feedback")
  if f["status"]!="pending": raise HTTPException(409,"Already reviewed")
  label=payload.verified_label or f["suggested_label"] if payload.decision=="approve" else None
  if payload.decision=="approve" and label not in classes(): raise HTTPException(422,"Verified label not in taxonomy")
  status="approved" if payload.decision=="approve" else "rejected"
  c.execute("INSERT INTO expert_reviews VALUES(?,?,?,?,?,?,?)",(str(uuid.uuid4()),fid,expert_id,payload.decision,label,payload.notes,now()))
  c.execute("UPDATE feedback SET status=? WHERE id=?",(status,fid))
  c.execute("INSERT INTO audit_log(actor_id,action,object_id,details,created_at) VALUES(?,?,?,?,?)",(expert_id,"feedback_"+status,fid,label or payload.notes,now()))
 return {"feedback_id":fid,"status":status,"verified_label":label}
@app.get("/api/expert/export.csv")
def export(expert_id:str=Header(alias="X-User-ID"),_=Depends(auth("expert"))):
 with db() as c: rows=c.execute("SELECT p.id,p.image_path,p.image_sha256,r.verified_label,r.expert_id,r.created_at FROM feedback f JOIN predictions p ON p.id=f.prediction_id JOIN expert_reviews r ON r.feedback_id=f.id WHERE f.status='approved' AND r.decision='approve' ORDER BY r.created_at").fetchall()
 out=io.StringIO(); w=csv.writer(out); w.writerow(["prediction_id","image_path","image_sha256","disease_label","expert_id","verified_at"])
 seen=set()
 for x in rows:
  if x["image_sha256"] not in seen: w.writerow([x["id"],x["image_path"],x["image_sha256"],x["verified_label"],x["expert_id"],x["created_at"]]); seen.add(x["image_sha256"])
 return Response(out.getvalue(),media_type="text/csv",headers={"Content-Disposition":'attachment; filename="verified_feedback.csv"'})
