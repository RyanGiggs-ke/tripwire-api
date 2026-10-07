"""Tripwire scoring API. Run: uvicorn main:app --reload"""
import os, joblib, pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

API_KEY = os.environ.get("TRIPWIRE_API_KEY", "dev-key")      # set a real secret in production
ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
model = joblib.load("model.joblib")                           # created by train.py

app = FastAPI(title="Tripwire fraud scoring API")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["POST", "GET"], allow_headers=["*"])

class Txn(BaseModel):
    type: str = Field(pattern="^(PAYMENT|TRANSFER|CASH_OUT|CASH_IN|DEBIT)$")
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    sender_balance_before: float = Field(ge=0)
    recipient_balance_before: float = Field(ge=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(t: Txn, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    row = pd.DataFrame([{
        "amount": t.amount, "hour": t.hour,
        "orig_drain": t.amount / (t.sender_balance_before + 1),
        "dest_zero_before": int(t.recipient_balance_before == 0),
        "type_code": TYPES.index(t.type)}])
    prob = float(model.predict_proba(row)[0, 1])
    s = round(prob * 100)
    reasons = []
    if row.orig_drain[0] > 0.9: reasons.append("Empties most of the sender's balance")
    if row.dest_zero_before[0]: reasons.append("Recipient account had no prior balance")
    if t.hour < 5 or t.hour >= 23: reasons.append("Unusual hour")
    return {"score": s, "decision": "block" if s >= 70 else "review" if s >= 35 else "approve", "reasons": reasons}
