"""Canonical Multi-OCR Vertex Adapter & Confidence-Gated Hybrid FastAPI Service."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from app.aadhaar_masker import OCRToken, compute_first8_mask_boxes


def normalize_quad_to_bbox(
    vertices: list[tuple[float, float]], img_w: int, img_h: int, already_normalized: bool = False
) -> tuple[float, float, float, float]:
    """Convert 4-point pixel or normalized OCR polygon into canonical [0..1] bbox."""
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    if already_normalized:
        return (
            round(max(0.0, min(xs)), 4),
            round(max(0.0, min(ys)), 4),
            round(min(1.0, max(xs)), 4),
            round(min(1.0, max(ys)), 4),
        )
    return (
        round(max(0.0, min(xs) / img_w), 4),
        round(max(0.0, min(ys) / img_h), 4),
        round(min(1.0, max(xs) / img_w), 4),
        round(min(1.0, max(ys) / img_h), 4),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = True
    app.state.ocr_engines = ("google_docai", "google_vision", "paddle_ocr", "azure_ocr")
    yield
    app.state.ready = False


app = FastAPI(title="Enterprise IDP & Hybrid OCR Service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - start) * 1000, 2))
    return response


class TokenInput(BaseModel):
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float = Field(ge=0.0, le=1.0)


class HybridExtractRequest(BaseModel):
    client_id: str = Field(min_length=2, max_length=40)
    doc_type: Literal["AADHAAR", "PAN", "AOF", "LOAN_PACKET", "CHEQUE", "DEMAND_DRAFT"]
    ocr_engine: Literal["google_docai", "google_vision", "paddle_ocr", "azure_ocr"] = "paddle_ocr"
    tokens: list[TokenInput]
    confidence_gate: float = Field(default=0.85, ge=0.5, le=1.0)


class HybridExtractResponse(BaseModel):
    client_id: str
    doc_type: str
    execution_tier: Literal["TIER_1_FAST_OCR", "TIER_2_HYBRID_VLM_FALLBACK"]
    mean_confidence: float
    aadhaar_mask_regions: list[tuple[int, int, int, int]]


@app.get("/health/ready")
async def readiness_probe(request: Request):
    return {"ready": bool(getattr(request.app.state, "ready", False))}


@app.post("/api/v1/idp/extract-and-mask", response_model=HybridExtractResponse)
async def extract_and_mask(payload: HybridExtractRequest):
    tokens = [OCRToken(t.text, t.bbox, t.confidence) for t in payload.tokens]
    mean_conf = round(sum(t.confidence for t in tokens) / max(len(tokens), 1), 4)
    tier = "TIER_1_FAST_OCR" if mean_conf >= payload.confidence_gate else "TIER_2_HYBRID_VLM_FALLBACK"
    mask_regions = compute_first8_mask_boxes(tokens) if payload.doc_type in ("AADHAAR", "AOF", "LOAN_PACKET") else []

    return HybridExtractResponse(
        client_id=payload.client_id.strip().upper(),
        doc_type=payload.doc_type,
        execution_tier=tier,
        mean_confidence=mean_conf,
        aadhaar_mask_regions=mask_regions,
    )
