# from fastapi import FastAPI

# app = FastAPI(
#     title="NeuroPilot API",
#     description="Alzheimer's Disease Detection using EEG and MRI",
#     version="1.0.0"
# )

# @app.get("/")
# def home():
#     return {
#         "message": "Welcome to NeuroPilot API"
#     }

# @app.get("/health")
# def health():
#     return {
#         "status": "Healthy"
#     }

# from fastapi import FastAPI
# from api.routes import router

# app = FastAPI(
#     title="NeuroPilot API",
#     version="1.0.0"
# )

# app.include_router(router)

# @app.get("/")
# def home():
#     return {"message": "Welcome to NeuroPilot API"}

# @app.get("/health")
# def health():
#     return {"status": "Healthy"}

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://127.0.0.1:5173",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include your routes
# app.include_router(router)
# @app.get("/")
# def home():
#     return {"message": "Welcome to NeuroPilot API"}

# @app.get("/health")
# def health():
#     return {"status": "Healthy"}

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

from database.database import engine
from database.models import Base
from api.history import router as history_router
from api.explain import router as explain_router
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="NeuroPilot API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes
app.include_router(router)
app.include_router(history_router)
app.include_router(explain_router)
@app.get("/")
def home():
    return {"message": "Welcome to NeuroPilot API"}

@app.get("/health")
def health():
    return {"status": "Healthy"}

from fastapi.routing import APIRoute

print("\n===== Registered Routes =====")
for route in app.routes:
    if isinstance(route, APIRoute):
        print(route.path)
print("=============================")