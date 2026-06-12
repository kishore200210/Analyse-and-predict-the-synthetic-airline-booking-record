from fastapi import FastAPI
from pydantic import BaseModel

from app import classify_with_groq, rule_label

app = FastAPI(title="Complaint Summary API")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


class ComplaintRequest(BaseModel):
    text: str


class ComplaintResponse(BaseModel):
    complaint: str
    keyword_category: str
    llm_category: str
    summary: str


@app.post("/classify", response_model=ComplaintResponse)
async def classify_complaint(payload: ComplaintRequest):
    keyword_cat = rule_label(payload.text)
    llm_cat = classify_with_groq(payload.text)

    summary = (
        f"The complaint is primarily classified as '{llm_cat}'. "
        f"Keyword rules also suggest '{keyword_cat}'. "
        "This gives a quick support-team view of the issue type."
    )

    return ComplaintResponse(
        complaint=payload.text,
        keyword_category=keyword_cat,
        llm_category=llm_cat,
        summary=summary,
    )
