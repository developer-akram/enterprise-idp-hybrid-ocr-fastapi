"""Verhoeff checksum validation and Aadhaar PII bounding-box masking logic."""

from dataclasses import dataclass

_D_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_P_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def is_valid_verhoeff_aadhaar(candidate: str) -> bool:
    """Validate a 12-digit Aadhaar candidate using UIDAI prefix and Verhoeff checksum."""
    digits = "".join(ch for ch in candidate if ch.isdigit())
    if len(digits) != 12 or digits[0] in ("0", "1"):
        return False
    checksum = 0
    for idx, ch in enumerate(reversed(digits)):
        checksum = _D_TABLE[checksum][_P_TABLE[idx % 8][int(ch)]]
    return checksum == 0


@dataclass(frozen=True)
class OCRToken:
    text: str
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float


def compute_first8_mask_boxes(tokens: list[OCRToken]) -> list[tuple[int, int, int, int]]:
    """
    Identify ALL valid Aadhaar occurrences on a page and return pixel boxes
    covering strictly the FIRST 8 DIGITS (preserving the last 4 digits for audit).
    """
    mask_boxes: list[tuple[int, int, int, int]] = []

    # Case 1: Single OCR token containing the full 12 digits (e.g., "2345 6789 0123")
    for tok in tokens:
        digits = "".join(ch for ch in tok.text if ch.isdigit())
        if len(digits) == 12 and is_valid_verhoeff_aadhaar(digits):
            x1, y1, x2, y2 = tok.bbox
            mask_x2 = int(x1 + (x2 - x1) * 0.67)
            mask_boxes.append((x1, y1, mask_x2, y2))

    # Case 2: Three collinear 4-digit tokens on the same horizontal Y-baseline (sorted left-to-right)
    four_digit = [t for t in tokens if len(t.text.strip()) == 4 and t.text.strip().isdigit()]
    four_digit.sort(key=lambda t: (t.bbox[1] + t.bbox[3]) / 2)
    lines: list[list[OCRToken]] = []
    for tok in four_digit:
        y_c = (tok.bbox[1] + tok.bbox[3]) / 2
        placed = False
        for line in lines:
            ref_y = (line[0].bbox[1] + line[0].bbox[3]) / 2
            line_tol = max(12.0, (line[0].bbox[3] - line[0].bbox[1]) * 0.6)
            if abs(y_c - ref_y) <= line_tol:
                line.append(tok)
                placed = True
                break
        if not placed:
            lines.append([tok])

    for line in lines:
        line.sort(key=lambda t: t.bbox[0])
        for i in range(len(line) - 2):
            t1, t2, t3 = line[i], line[i + 1], line[i + 2]
            combined = t1.text.strip() + t2.text.strip() + t3.text.strip()
            if is_valid_verhoeff_aadhaar(combined):
                mask_boxes.append(t1.bbox)
                mask_boxes.append(t2.bbox)

    return mask_boxes
