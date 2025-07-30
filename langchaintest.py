from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
bearer = HTTPBearer()

@app.post("/api/auth/login")
def receive_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    token = credentials.credentials
    print("Received JWT from Spring:", token)
    # 검증 로직 추가 가능
    return {"status": "ok", "token": token}
