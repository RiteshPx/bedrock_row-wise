"""
POC: PDF Paragraph Highlighter
-------------------------------
Flow:
  PDF
   -> PyMuPDF   : row-wise coordinates (per line/block)
   -> PaddleOCR : extract full text
   -> LLM       : given text + row coords, user query → return coords to highlight
   -> PyMuPDF   : draw yellow highlights on matched rows
   -> Save highlighted PDF

Usage:
  python main.py --pdf invoice.pdf --query "find the total amount"
"""

import os
import re
import fitz
import json
import boto3
import numpy as np
import argparse

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

# ─── CONFIG ───────────────────────────────────────────────────────────────────

REGION   = "us-east-1"
MODEL_ID = "us.anthropic.claude-opus-4-5-20251101-v1:0"

HIGHLIGHT_COLOR = (1, 1, 0)   # yellow (RGB 0-1)
HIGHLIGHT_ALPHA = 0.4
DPI             = 200

# ─── OCR SINGLETON ────────────────────────────────────────────────────────────

_ocr = None
def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr

# ─── STEP 1: PyMuPDF → row-wise coordinates ───────────────────────────────────

def extract_rows(pdf_path: str) -> list:
    """
    Extract text lines with bounding boxes using PyMuPDF.

    Returns list of:
        {"row_id": 0, "page": 0, "x0":..., "y0":..., "x1":..., "y1":..., "text": "..."}

    Uses dict-based extraction for reliable line-level grouping.
    """
    doc  = fitz.open(pdf_path)
    rows = []
    row_id = 0

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:   # 0 = text block
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                # Merge all spans in a line → one row
                text = " ".join(s["text"].strip() for s in spans if s["text"].strip())
                if not text:
                    continue

                x0 = min(s["bbox"][0] for s in spans)
                y0 = min(s["bbox"][1] for s in spans)
                x1 = max(s["bbox"][2] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)

                rows.append({
                    "row_id": row_id,
                    "page"  : page_num,
                    "x0"    : round(x0, 2),
                    "y0"    : round(y0, 2),
                    "x1"    : round(x1, 2),
                    "y1"    : round(y1, 2),
                    "text"  : text
                })
                row_id += 1

    doc.close()
    print("rows extracted: ", rows)
    print("number of rows: ", len(rows))
    return rows


# ─── STEP 2: PaddleOCR → full text (for scanned or to cross-verify) ───────────

def extract_text_ocr(pdf_path: str) -> str:
    """
    Extract full text from PDF using PaddleOCR.
    Renders each page to image, runs OCR, returns combined text.
    """
    doc   = fitz.open(pdf_path)
    ocr   = get_ocr()
    pages_text = []

    for page in doc:
        mat    = fitz.Matrix(DPI / 72, DPI / 72)
        pix    = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img    = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

        try:
            result = ocr.ocr(img, cls=True)
        except TypeError:
            result = ocr.ocr(img)

        lines = []
        if result and result[0]:
            for line in result[0]:
                _, (text, _) = line
                if text.strip():
                    lines.append(text.strip())
        pages_text.append("\n".join(lines))

    doc.close()
    return "\n\n".join(pages_text)


# ─── STEP 3: LLM call → returns row_ids to highlight ──────────────────────────

def ask_llm_which_rows(rows: list, ocr_text: str, user_query: str) -> list:
    """
    Send rows + OCR text + user query to Claude.
    LLM returns a list of row_ids that are relevant to the query.

    Returns: [0, 3, 4, 5, ...]
    """
    client = boto3.client("bedrock-runtime", region_name=REGION)

    # Compact row summary for the prompt (row_id + text only)
    row_summary = "\n".join(
        f'[{r["row_id"]}] {r["text"]}' for r in rows
    )

    prompt = f"""You are a document analysis assistant.

User query: "{user_query}"

Below are numbered rows extracted from a PDF document:
{row_summary}

OCR extracted text (for context):
{ocr_text[:3000]}

Task:
- Identify which rows are relevant to the user query.
- If consecutive rows form a paragraph/section about the query, include all of them.
- Return ONLY a JSON array of row_id integers. Nothing else. No explanation.

Example output: [2, 3, 4, 7]
"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens"       : 512,
        "temperature"      : 0,
        "messages"         : [{"role": "user", "content": prompt}]
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    raw = json.loads(response["body"].read())["content"][0]["text"].strip()
    print("========================================================================================")
    print("raw: ", raw)
    print("========================================================================================")
    # Parse JSON array from response (strip markdown fences if present)
    raw = re.sub(r"```[a-z]*", "", raw).strip("` \n")
    print("========================================================================================")
    print("raw: ", raw)
    print("========================================================================================")
    row_ids = json.loads(raw)
    print("========================================================================================")
    print("row_ids: ", row_ids)
    print("========================================================================================")
    print("number of row_ids: ", len(row_ids))
    print("========================================================================================")
    return row_ids


# ─── STEP 4: PyMuPDF → highlight matched rows ────────────────────────────────

def highlight_rows(pdf_path: str, rows: list, row_ids: list, output_path: str):
    """
    Draw yellow highlight annotations on matched rows.
    Merges consecutive rows on the same page into one rect.
    """
    doc = fitz.open(pdf_path)

    # Group matched rows by page
    matched = [r for r in rows if r["row_id"] in set(row_ids)]

    # Sort by page then y0 so we can merge consecutive rows
    matched.sort(key=lambda r: (r["page"], r["y0"]))

    # Merge rows that are close vertically (same paragraph)
    def merge_close_rows(page_rows, gap_threshold=5):
        if not page_rows:
            return []
        merged = []
        cur = dict(page_rows[0])
        for r in page_rows[1:]:
            if r["y0"] - cur["y1"] <= gap_threshold:
                # Extend current rect
                cur["x0"] = min(cur["x0"], r["x0"])
                cur["x1"] = max(cur["x1"], r["x1"])
                cur["y1"] = r["y1"]
            else:
                merged.append(cur)
                cur = dict(r)
        merged.append(cur)
        return merged

    # Group by page and merge
    from itertools import groupby
    for page_num, group in groupby(matched, key=lambda r: r["page"]):
        page_rows  = list(group)
        page       = doc[page_num]
        merged     = merge_close_rows(page_rows)

        for rect_data in merged:
            rect  = fitz.Rect(
                rect_data["x0"] - 2,
                rect_data["y0"] - 2,
                rect_data["x1"] + 2,
                rect_data["y1"] + 2
            )
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=HIGHLIGHT_COLOR)
            annot.set_opacity(HIGHLIGHT_ALPHA)
            annot.update()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"✅ Highlighted PDF saved: {output_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(pdf_path: str, user_query: str, output_path: str = None):
    if output_path is None:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"outputs/{base}_highlighted.pdf"

    print(f"\n📄 PDF     : {pdf_path}")
    print(f"🔍 Query   : {user_query}")
    print(f"💾 Output  : {output_path}\n")

    print("[1/4] Extracting rows with PyMuPDF...")
    rows = extract_rows(pdf_path)
    print(f"      {len(rows)} rows found")

    print("[2/4] Extracting text with PaddleOCR...")
    ocr_text = extract_text_ocr(pdf_path)
    print(f"      {len(ocr_text)} characters extracted")

    print("[3/4] Asking LLM which rows to highlight...")
    row_ids = ask_llm_which_rows(rows, ocr_text, user_query)
    print(f"      Row IDs to highlight: {row_ids}")

    print("[4/4] Drawing highlights on PDF...")
    highlight_rows(pdf_path, rows, row_ids, output_path)

    return {"highlighted_pdf": output_path, "highlighted_row_ids": row_ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Paragraph Highlighter")
    parser.add_argument("--pdf",   required=True, help="Path to input PDF")
    parser.add_argument("--query", required=True, help="What to highlight")
    parser.add_argument("--out",   default=None,  help="Output PDF path")
    args = parser.parse_args()

    run(args.pdf, args.query, args.out)