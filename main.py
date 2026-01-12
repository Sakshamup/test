from fastapi import FastAPI
from route import router

app = FastAPI(title="Veterinary EMR Analyzer")

app.include_router(router)
