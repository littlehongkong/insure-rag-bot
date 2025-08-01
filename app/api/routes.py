from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.rag_pipeline import RAGPipeline
from app.core.insurance_analyzer import InsuranceAnalyzer

router = APIRouter()

rag_pipeline = RAGPipeline()
analyzer = InsuranceAnalyzer()

@router.post("/upload")
async def upload_policy(file: UploadFile = File(...)):
    """Upload insurance policy PDF and process it"""
    try:
        # TODO: Implement PDF processing
        return {"message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_policy(policy_id: str, query: str):
    """Analyze uploaded policy with a specific query"""
    try:
        # TODO: Implement policy analysis
        return {"message": "Analysis completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare_policies(policy_a: str, policy_b: str):
    """Compare two insurance policies"""
    try:
        # TODO: Implement policy comparison
        return {"message": "Comparison completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
