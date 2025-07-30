from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
bearer = HTTPBearer()

@app.post("/api/auth/login")
def receive_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    token = credentials.credentials
    # TODO: 디코딩/검증 로직 또는 다른 처리
    return {"status": "received", "token": token}
