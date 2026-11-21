# Enterprise IDP & Hybrid OCR Microservice (FastAPI + Pydantic v2 + GKE)

A production-oriented **Intelligent Document Processing (IDP)** architecture demonstrating key patterns used in high-volume BFSI KYC automation:

- **Canonical Multi-OCR Adapter:** Normalizes disparate 4-point pixel polygons (Google Cloud Vision / PaddleOCR) and `0.0–1.0` normalized vertices (Google Document AI / Azure) into a single canonical `[x_min, y_min, x_max, y_max]` coordinate schema.
- **Fail-Closed Aadhaar PII Masking Engine:**
  - Validates 12-digit candidates using the **Verhoeff Checksum algorithm** + UIDAI prefix rules (`first_digit >= 2`) so 12-digit Bank Account / AOF numbers are never falsely masked.
  - Supports both **multi-token 4-digit collinear grouping** (`Box1`, `Box2`, `Box3`) and **single-box proportional 67% width slicing** to mask strictly the **first 8 digits** (`XXXX XXXX 9012`) while keeping the last 4 digits readable for audit verification.
- **Confidence-Gated 2-Tier Hybrid Extraction:** Routes high-confidence (`>= 0.85`) documents through fast Tier-1 deterministic rules (`<400ms`), escalating only degraded/complex layouts to a Pydantic-enforced VLM/LLM schema fallback.
- **Cloud-Native GKE & Async Architecture:** Includes FastAPI `lifespan` startup/shutdown hooks, `X-Correlation-ID` observability middleware, `Dockerfile`, and Kubernetes (`GKE`) Deployment YAML with `readinessProbe`, `maxUnavailable: 0` rolling updates, and memory limits.

---

## Quick Start & Tests

```bash
pip install -r requirements.txt
pytest -v
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
