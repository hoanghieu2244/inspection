from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .retriever import Retriever, Document
from .evaluator import generate_report

app = FastAPI(title="Evidence Finder & Report Generator (prototype)")

# in-memory retriever instance (simple prototype)
retriever = Retriever()

class IndexRequest(BaseModel):
    documents: List[Document]

class ReportRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.post("/index")
async def index(req: IndexRequest):
    try:
        retriever.index_documents(req.documents)
        # optionally save docs JSONL
        retriever.save_documents_jsonl(req.documents)
        return {"status": "ok", "count": len(req.documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report")
async def report(req: ReportRequest):
    try:
        evidences = retriever.retrieve(req.query, top_k=req.top_k)
        result = generate_report(req.query, evidences)
        # add note about weak evidence if scores low
        top_score = max([e["score"] for e in evidences], default=0.0)
        if top_score < 0.4:
            result["_note"] = "Minh chứng yếu (top score < 0.4). Recommend human review."
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
