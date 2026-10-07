# Tripwire API
1. pip install -r requirements.txt
2. python train.py            (or: python train.py paysim.csv)
3. uvicorn main:app --reload
4. Test:
   curl -X POST localhost:8000/score -H "x-api-key: dev-key" -H "Content-Type: application/json" \
   -d '{"type":"TRANSFER","amount":145000,"hour":3,"sender_balance_before":146000,"recipient_balance_before":0}'
Deploy: Render, Railway or Fly.io. Set TRIPWIRE_API_KEY and ALLOWED_ORIGINS.
