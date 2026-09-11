import os, base64, json, re, io
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image

app = FastAPI(title="Fantasy Command Vision Service", version="28.8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

MODEL = os.getenv("FANTASY_COMMAND_VISION_MODEL", "gpt-5.6-terra")

def configured():
    return bool(os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Fantasy Command V28.8 Vision Service",
        "health": "/api/health"
    }

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "Fantasy Command V28.8 Vision Service",
        "version": "V28.8",
        "vision_configured": configured(),
        "model": MODEL if configured() else None,
        "phone_board_reader": True,
        "column_slice_reader": True,
        "cell_reader": True
    }

def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise

def data_url(raw: bytes, mime="image/jpeg"):
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")

def snake_pick(slot, round_num, teams):
    if round_num % 2:
        return (round_num - 1) * teams + slot
    return round_num * teams - slot + 1

@app.post("/api/vision/cell")
async def vision_cell(
    image: UploadFile = File(...),
    tournament_id: str = Form(...),
    platform: str = Form(...),
    field_size: int = Form(...),
    team_slot: int = Form(...),
    round: int = Form(...),
    pick: int = Form(...),
    current_read: str = Form("")
):
    if not configured():
        raise HTTPException(
            503,
            "Hosted vision service is online but OPENAI_API_KEY is not configured."
        )

    raw = await image.read()

    if not raw:
        raise HTTPException(400, "Empty cell image.")

    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(413, "Cell image is too large.")

    try:
        im = Image.open(io.BytesIO(raw))
        im.load()

        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")

        out = io.BytesIO()
        im.save(out, format="JPEG", quality=96, optimize=True)
        raw = out.getvalue()

    except Exception as e:
        raise HTTPException(
            400,
            f"Could not decode cell image: {type(e).__name__}"
        )

    prompt = f"""
You are Fantasy Command V28.8 CELL READER.

The attached image is one enlarged fantasy-football draft-board cell only.

Platform: {platform}
Team slot: {team_slot}
Round: {round}
Overall pick: {pick}
Previous OCR read: {current_read or 'none'}

Read ONLY the player name and displayed fantasy position in this single cell.

Do not infer from draft rankings, ADP, the previous OCR read, or nearby expected players.
Preserve what is visibly printed.

If unreadable, return an empty name and explain in visible_text.

Return ONLY JSON:

{{
  "name": "player name or empty",
  "pos": "QB|RB|WR|TE|DST|K or empty",
  "confidence": 0.0,
  "visible_text": "brief literal text you can see"
}}
"""

    try:
        client = OpenAI()

        resp = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": data_url(raw),
                            "detail": "high"
                        }
                    ]
                }
            ]
        )

        d = extract_json(resp.output_text)

        name = str(d.get("name") or "").strip()
        pos = str(d.get("pos") or d.get("position") or "").upper().strip()

        if pos in ("DEF", "D/ST"):
            pos = "DST"

        if pos not in ("QB", "RB", "WR", "TE", "DST", "K"):
            pos = ""

        return {
            "name": name,
            "pos": pos,
            "confidence": d.get("confidence"),
            "visible_text": str(d.get("visible_text") or ""),
            "team_slot": team_slot,
            "round": round,
            "pick": pick,
            "reader_model": MODEL
        }

    except HTTPException:
        raise

    except Exception as e:
        msg = str(e)

        if (
            "429" in msg
            or "credits" in msg.lower()
            or "quota" in msg.lower()
        ):
            raise HTTPException(
                429,
                "OpenAI API credits are unavailable."
            )

        raise HTTPException(
            500,
            f"Cell vision processing failed: {type(e).__name__}: {msg[:300]}"
        )
