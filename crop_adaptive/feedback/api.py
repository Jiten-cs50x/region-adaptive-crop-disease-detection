from __future__ import annotations
import csv, hashlib, io, json, os, secrets, sqlite3, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

ROOT=Path(os.getenv("CROP_FEEDBACK_HOME","runs/feedback")).resolve()
UPLOADS=ROOT/"uploads"; DB_PATH=ROOT/"feedback.sqlite3"
ROOT.mkdir(parents=True,exist_ok=True); UPLOADS.mkdir(parents=True,exist_ok=True)
app=FastAPI(title="Agrivision AI API",version="1.1.0")
SESSION_COOKIE="agrivision_session"
SESSION_MAX_AGE=12*60*60
def session_serializer():
 secret=os.getenv("CROP_SESSION_SECRET")
 if not secret or len(secret)<32: raise HTTPException(503,"Set CROP_SESSION_SECRET to a random value of at least 32 characters")
 return URLSafeTimedSerializer(secret,salt="agrivision-auth-v1")
def hash_password(password,salt=None):
 salt=salt or secrets.token_bytes(16)
 digest=hashlib.pbkdf2_hmac("sha256",password.encode(),salt,310000)
 return salt.hex()+"$"+digest.hex()
def verify_password(password,stored):
 try:
  salt_hex,digest_hex=stored.split("$",1)
  return secrets.compare_digest(hash_password(password,bytes.fromhex(salt_hex)).split("$",1)[1],digest_hex)
 except (ValueError,TypeError): return False
def ensure_auth_schema():
 with db() as c:
  c.execute("""CREATE TABLE IF NOT EXISTS users(
   id TEXT PRIMARY KEY,username TEXT NOT NULL UNIQUE,display_name TEXT NOT NULL,
   phone TEXT UNIQUE,password_hash TEXT NOT NULL,role TEXT NOT NULL CHECK(role IN ('farmer','expert','admin')),
   active INTEGER NOT NULL DEFAULT 1,session_version INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL)""")
  admin_name=os.getenv("AGRIVISION_ADMIN_USERNAME","").strip()
  admin_password=os.getenv("AGRIVISION_ADMIN_PASSWORD","")
  if admin_name and admin_password and not c.execute("SELECT id FROM users WHERE username=?",(admin_name,)).fetchone():
   c.execute("INSERT INTO users(id,username,display_name,password_hash,role,created_at) VALUES(?,?,?,?,?,?)",
    (str(uuid.uuid4()),admin_name,"Agrivision Administrator",hash_password(admin_password),"admin",now()))
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
 CREATE UNIQUE INDEX IF NOT EXISTS expert_reviews_feedback_unique ON expert_reviews(feedback_id);
 """)
@app.on_event("startup")
def initialize_auth():
 ensure_auth_schema()
class RegisterFarmer(BaseModel):
 username:str=Field(min_length=3,max_length=40,pattern=r"^[a-zA-Z0-9_.-]+$")
 display_name:str=Field(min_length=2,max_length=100)
 phone:Optional[str]=Field(default=None,max_length=30)
 password:str=Field(min_length=10,max_length=128)
class Login(BaseModel):
 username:str=Field(min_length=1,max_length=40)
 password:str=Field(min_length=1,max_length=128)
class CreateExpert(BaseModel):
 username:str=Field(min_length=3,max_length=40,pattern=r"^[a-zA-Z0-9_.-]+$")
 display_name:str=Field(min_length=2,max_length=100)
 password:str=Field(min_length=10,max_length=128)
def issue_session(response,user):
 csrf=secrets.token_urlsafe(32)
 token=session_serializer().dumps({"uid":user["id"],"role":user["role"],"ver":user["session_version"],"csrf":csrf})
 response.set_cookie(SESSION_COOKIE,token,max_age=SESSION_MAX_AGE,httponly=True,
  secure=os.getenv("CROP_COOKIE_SECURE","0")=="1",samesite="lax",path="/")
 response.headers["X-CSRF-Token"]=csrf
 return response
def auth(role):
 async def check(request:Request):
  token=request.cookies.get(SESSION_COOKIE)
  if not token: raise HTTPException(401,"Please sign in")
  try: data=session_serializer().loads(token,max_age=SESSION_MAX_AGE)
  except (BadSignature,SignatureExpired): raise HTTPException(401,"Session expired. Please sign in again")
  with db() as c: user=c.execute("SELECT id,role,active,session_version FROM users WHERE id=?",(data.get("uid"),)).fetchone()
  if not user or not user["active"] or user["role"]!=data.get("role") or user["session_version"]!=data.get("ver"):
   raise HTTPException(401,"Account session is no longer valid")
  if role!="any" and user["role"]!=role: raise HTTPException(403,"This account cannot access that workspace")
  if request.method in {"POST","PUT","PATCH","DELETE"}:
   supplied=request.headers.get("X-CSRF-Token","")
   if not supplied or not secrets.compare_digest(supplied,str(data.get("csrf",""))):
    raise HTTPException(403,"CSRF validation failed")
  return user["id"]
 return check
@app.post("/api/auth/register")
def register(payload:RegisterFarmer):
 ensure_auth_schema()
 with db() as c:
  try:
   uid=str(uuid.uuid4())
   c.execute("INSERT INTO users(id,username,display_name,phone,password_hash,role,created_at) VALUES(?,?,?,?,?,'farmer',?)",
    (uid,payload.username,payload.display_name,payload.phone,hash_password(payload.password),now()))
  except sqlite3.IntegrityError: raise HTTPException(409,"Username or phone number is already registered")
  user=c.execute("SELECT * FROM users WHERE id=?",(uid,)).fetchone()
  from fastapi.responses import JSONResponse
  response=JSONResponse({"id":uid,"username":user["username"],"display_name":user["display_name"],"role":"farmer"})
  return issue_session(response,user)
@app.post("/api/auth/login")
def login(payload:Login):
 ensure_auth_schema()
 with db() as c: user=c.execute("SELECT * FROM users WHERE username=?",(payload.username,)).fetchone()
 if not user or not user["active"] or not verify_password(payload.password,user["password_hash"]):
  raise HTTPException(401,"Incorrect username or password")
 from fastapi.responses import JSONResponse
 response=JSONResponse({"id":user["id"],"username":user["username"],"display_name":user["display_name"],"role":user["role"]})
 return issue_session(response,user)
@app.get("/api/auth/me")
def me(user_id:str=Depends(auth("any"))):
 with db() as c: user=c.execute("SELECT id,username,display_name,role FROM users WHERE id=?",(user_id,)).fetchone()
 return dict(user)
@app.post("/api/auth/logout")
def logout(user_id:str=Depends(auth("any"))):
 with db() as c: c.execute("UPDATE users SET session_version=session_version+1 WHERE id=?",(user_id,))
 from fastapi.responses import JSONResponse
 response=JSONResponse({"status":"signed_out"})
 response.delete_cookie(SESSION_COOKIE,path="/")
 return response
@app.post("/api/admin/experts")
def create_expert(payload:CreateExpert,admin_id:str=Depends(auth("admin"))):
 with db() as c:
  try:
   uid=str(uuid.uuid4())
   c.execute("INSERT INTO users(id,username,display_name,password_hash,role,created_at) VALUES(?,?,?,?,'expert',?)",
    (uid,payload.username,payload.display_name,hash_password(payload.password),now()))
  except sqlite3.IntegrityError: raise HTTPException(409,"Username is already registered")
  c.execute("INSERT INTO audit_log(actor_id,action,object_id,details,created_at) VALUES(?,?,?,?,?)",
   (admin_id,"expert_account_created",uid,payload.username,now()))
 return {"id":uid,"username":payload.username,"display_name":payload.display_name,"role":"expert"}
@app.get("/api/admin/experts")
def list_experts(admin_id:str=Depends(auth("admin"))):
 with db() as c: rows=c.execute("SELECT id,username,display_name,active,created_at FROM users WHERE role='expert' ORDER BY created_at DESC").fetchall()
 return [dict(x) for x in rows]
def classes():
 import yaml
 p=Path(os.getenv("CROP_CONFIG","configs/sugarcane_karnataka.yaml"))
 if not p.is_file(): raise HTTPException(503,"CROP_CONFIG taxonomy file not found")
 return yaml.safe_load(p.read_text(encoding="utf-8"))["taxonomy"]["classes"]
def predictor():
 p=Path(os.getenv("CROP_MODEL_CHECKPOINT",""))
 if not p.is_file(): raise HTTPException(503,"Set CROP_MODEL_CHECKPOINT to enable inference")
 config_path=Path(os.getenv("CROP_CONFIG","configs/sugarcane_karnataka.yaml"))
 if not config_path.is_file(): raise HTTPException(503,f"CROP_CONFIG not found: {config_path}")
 try:
  import torch
  import yaml
  from crop_adaptive.data.transforms import build_transforms
  from crop_adaptive.models.efficientnet import build_model_from_config
  config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
  labels=config.get("taxonomy",{}).get("classes",[])
  if not isinstance(labels,list) or len(labels)<2: raise ValueError("Config must define taxonomy.classes")
  try: ckpt=torch.load(p,map_location="cpu",weights_only=False)
  except TypeError: ckpt=torch.load(p,map_location="cpu")
  mapping={name:index for index,name in enumerate(labels)}
  if ckpt.get("classes")!=labels or ckpt.get("class_to_idx")!=mapping:
   raise ValueError("Checkpoint class order/mapping does not match CROP_CONFIG taxonomy.classes")
  model=build_model_from_config(config,pretrained_override=False)
  model.load_state_dict(ckpt["model_state_dict"]); model.eval()
  tf=build_transforms(config,"predict")
  return model,labels,tf,hashlib.sha256(p.read_bytes()).hexdigest()[:12],torch
 except Exception as e: raise HTTPException(500,f"Checkpoint load failed: {e}") from e
class Correction(BaseModel):
 suggested_label:str=Field(min_length=1,max_length=100); notes:str=Field(default="",max_length=2000)
class Review(BaseModel):
 decision:str=Field(pattern="^(approve|reject)$"); verified_label:Optional[str]=None; notes:str=Field(default="",max_length=2000)
@app.get("/health")
def health(): return {"status":"ok","database":str(DB_PATH)}
@app.get("/api/taxonomy")
def taxonomy(): return {"classes":classes()}
@app.get("/",response_class=HTMLResponse)
def home(): return HTMLResponse((Path(__file__).with_name("dashboard.html")).read_text(encoding="utf-8"))
@app.post("/api/predictions")
async def predict(image:UploadFile=File(...),farmer_id:str=Depends(auth("farmer"))):
 if image.content_type not in {"image/jpeg","image/png","image/webp"}: raise HTTPException(415,"Upload JPEG, PNG or WebP")
 raw=await image.read(10*1024*1024+1)
 if len(raw)>10*1024*1024: raise HTTPException(413,"Image exceeds 10 MB")
 try:
  with Image.open(io.BytesIO(raw)) as im: im.verify()
  with Image.open(io.BytesIO(raw)) as im: rgb=ImageOps.exif_transpose(im).convert("RGB")
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
def correct(pid:str,payload:Correction,farmer_id:str=Depends(auth("farmer"))):
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
def mine(farmer_id:str=Depends(auth("farmer"))):
 with db() as c:
  rows=c.execute("""SELECT p.id,p.predicted_label,p.confidence,p.created_at,f.suggested_label,f.status,
   r.verified_label,r.notes AS expert_notes,r.expert_id,r.created_at AS reviewed_at,r.decision
   FROM predictions p LEFT JOIN feedback f ON f.prediction_id=p.id
   LEFT JOIN expert_reviews r ON r.feedback_id=f.id
   WHERE p.farmer_id=? ORDER BY p.created_at DESC""",(farmer_id,)).fetchall()
 return [dict(x) for x in rows]
@app.get("/api/expert/queue")
def queue(expert_id:str=Depends(auth("expert"))):
 with db() as c: rows=c.execute("SELECT f.id feedback_id,f.prediction_id,f.farmer_id,f.suggested_label,f.notes,p.predicted_label,p.confidence FROM feedback f JOIN predictions p ON p.id=f.prediction_id WHERE f.status='pending' ORDER BY f.created_at").fetchall()
 return [dict(x) for x in rows]
@app.get("/api/expert/image/{pid}")
def expert_image(pid:str,expert_id:str=Depends(auth("expert"))):
 with db() as c: p=c.execute("SELECT image_path FROM predictions WHERE id=?",(pid,)).fetchone()
 if not p or not Path(p["image_path"]).is_file(): raise HTTPException(404,"Image not found")
 return FileResponse(p["image_path"])
@app.post("/api/expert/review/{fid}")
def review(fid:str,payload:Review,expert_id:str=Depends(auth("expert"))):
 with db() as c:
  c.execute("BEGIN IMMEDIATE")
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
def export(expert_id:str=Depends(auth("expert"))):
 with db() as c: rows=c.execute("SELECT p.id,p.image_sha256,r.verified_label,r.expert_id,r.created_at,r.notes FROM feedback f JOIN predictions p ON p.id=f.prediction_id JOIN expert_reviews r ON r.feedback_id=f.id WHERE f.status='approved' AND r.decision='approve' ORDER BY r.created_at").fetchall()
 out=io.StringIO(); w=csv.writer(out); w.writerow(["prediction_id","image_sha256","disease_label","expert_id","verified_at","expert_notes"])
 seen=set()
 for x in rows:
  if x["image_sha256"] not in seen:
   w.writerow([x["id"],x["image_sha256"],x["verified_label"],x["expert_id"],x["created_at"],x["notes"]]); seen.add(x["image_sha256"])
 return Response(out.getvalue(),media_type="text/csv",headers={"Content-Disposition":'attachment; filename="verified_feedback.csv"'})

@app.get("/api/expert/reviews")
def review_history(expert_id:str=Depends(auth("expert"))):
 with db() as c:
  rows=c.execute("""SELECT f.id AS feedback_id,f.prediction_id,f.farmer_id,f.suggested_label,f.status,p.predicted_label,p.confidence,r.decision,r.verified_label,r.notes AS expert_notes,r.expert_id,r.created_at AS reviewed_at FROM feedback f JOIN predictions p ON p.id=f.prediction_id JOIN expert_reviews r ON r.feedback_id=f.id ORDER BY r.created_at DESC""").fetchall()
 return [dict(x) for x in rows]
