from fastapi.testclient import TestClient
from app.aadhaar_masker import OCRToken, compute_first8_mask_boxes, is_valid_verhoeff_aadhaar
from app.main import app, normalize_quad_to_bbox


def test_verhoeff_rejects_bank_account_and_accepts_valid_aadhaar():
    # Generate a known valid Verhoeff 12-digit number starting with >= 2
    # 20000000000 -> verify Verhoeff check digit
    valid_candidate = None
    for d in range(10):
        c = f"23456789012{d}"
        if is_valid_verhoeff_aadhaar(c):
            valid_candidate = c
            break
    assert valid_candidate is not None
    assert is_valid_verhoeff_aadhaar(valid_candidate) is True
    assert is_valid_verhoeff_aadhaar("012345678901") is False  # Invalid prefix '0'


def test_single_box_masks_only_first_67_percent():
    valid = next(f"23456789012{d}" for d in range(10) if is_valid_verhoeff_aadhaar(f"23456789012{d}"))
    tokens = [OCRToken(text=valid, bbox=(100, 200, 400, 240), confidence=0.95)]
    boxes = compute_first8_mask_boxes(tokens)
    assert len(boxes) == 1
    assert boxes[0] == (100, 200, 301, 240)  # First ~67% width masked, last 4 digits preserved


def test_collinear_three_box_aadhaar_masks_first_two_boxes():
    valid = next(f"34567890123{d}" for d in range(10) if is_valid_verhoeff_aadhaar(f"34567890123{d}"))
    tokens = [
        OCRToken(text=valid[:4], bbox=(50, 100, 120, 130), confidence=0.96),
        OCRToken(text=valid[4:8], bbox=(130, 102, 200, 132), confidence=0.94),
        OCRToken(text=valid[8:], bbox=(210, 101, 280, 131), confidence=0.97),
    ]
    boxes = compute_first8_mask_boxes(tokens)
    assert boxes == [(50, 100, 120, 130), (130, 102, 200, 132)]


def test_normalize_quad_and_fastapi_tier_routing():
    norm = normalize_quad_to_bbox([(100, 200), (500, 200), (500, 300), (100, 300)], img_w=1000, img_h=1000)
    assert norm == (0.1, 0.2, 0.5, 0.3)

    with TestClient(app) as client:
        r_ready = client.get("/health/ready")
        assert r_ready.status_code == 200
        assert r_ready.json()["ready"] is True

        r_low = client.post(
            "/api/v1/idp/extract-and-mask",
            json={
                "client_id": "icici_kyc",
                "doc_type": "AOF",
                "tokens": [{"text": "Account Opening", "bbox": [10, 10, 100, 30], "confidence": 0.62}],
            },
        )
        assert r_low.status_code == 200
        assert r_low.json()["execution_tier"] == "TIER_2_HYBRID_VLM_FALLBACK"
        assert "X-Correlation-ID" in r_low.headers
