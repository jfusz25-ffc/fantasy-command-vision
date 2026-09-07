import os, base64, json, re, io
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image

app = FastAPI(title="Fantasy Command Vision Service", version="25")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

MODEL = os.getenv("FANTASY_COMMAND_VISION_MODEL", "gpt-5.6-terra")

def configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"ok": True, "service": "Fantasy Command V25 Vision Service", "health": "/api/health"}

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "Fantasy Command V25 Vision Service",
        "version": "V25",
        "vision_configured": configured(),
        "model": MODEL if configured() else None,
        "phone_board_reader": True,
    }

def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
        raise

def image_data_url(img: Image.Image, quality: int = 92) -> str:
    out = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")

def prepare_views(raw: bytes) -> list[tuple[str,str]]:
    """Return full board plus overlapping vertical zoom bands for phone screenshots."""
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:
        raise HTTPException(400, f"Could not decode screenshot: {type(e).__name__}")
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size
    # Keep enough resolution for tiny roster text without making payload enormous.
    max_w = 2200
    if w > max_w:
        ratio = max_w / w
        im = im.resize((max_w, max(1, int(h * ratio))), Image.LANCZOS)
        w, h = im.size
    views = [("FULL BOARD", image_data_url(im))]
    # Overlapping thirds dramatically enlarge roster text for portrait phone boards.
    spans = [(0.00, 0.43, "TOP ZOOM"), (0.28, 0.72, "MIDDLE ZOOM"), (0.57, 1.00, "BOTTOM ZOOM")]
    for a,b,label in spans:
        y0, y1 = int(h*a), int(h*b)
        crop = im.crop((0, y0, w, y1))
        # Upscale narrow phone boards so each draft cell is easier to read.
        target_w = 2400 if crop.width < 2400 else crop.width
        if crop.width < target_w:
            ratio = target_w / crop.width
            crop = crop.resize((target_w, max(1, int(crop.height*ratio))), Image.LANCZOS)
        views.append((label, image_data_url(crop)))
    return views

def snake_pick(slot: int, round_num: int, teams: int) -> int:
    return (round_num - 1) * teams + slot if round_num % 2 == 1 else round_num * teams - slot + 1

def validate_result(data: dict, field_size: int, platform_hint: str) -> dict:
    expected_per_team = 20 if platform_hint.lower() == "draftkings" else 18
    teams = data.get("teams") or data.get("pod") or []
    if not isinstance(teams, list) or not teams:
        raise HTTPException(422, "Vision model returned no teams.")

    by_slot = {}
    review = []
    warnings = list(data.get("warnings") or [])
    for raw in data.get("review_cells") or []:
        try:
            slot = int(raw.get("team_slot") or raw.get("slot"))
            rnd = int(raw.get("round"))
        except Exception:
            continue
        if 1 <= slot <= field_size and 1 <= rnd <= expected_per_team:
            review.append({
                "team_slot": slot,
                "round": rnd,
                "visible_text": str(raw.get("visible_text") or raw.get("visibleText") or ""),
                "reason": str(raw.get("reason") or "Vision could not read this cell confidently."),
            })

    for i, t in enumerate(teams):
        try:
            slot = int(t.get("slot") or i+1)
        except Exception:
            continue
        if not (1 <= slot <= field_size) or slot in by_slot:
            continue
        round_map = {}
        for j,p in enumerate(t.get("players") or []):
            name = str(p.get("name") or "").strip()
            pos = str(p.get("pos") or p.get("position") or "").upper().strip()
            if pos in ("DST","DEF","D/ST"):
                pos = "DST"
            try:
                rnd = int(p.get("round") or j+1)
            except Exception:
                continue
            if not name or pos not in ("QB","RB","WR","TE","DST","K") or not (1 <= rnd <= expected_per_team) or rnd in round_map:
                continue
            try:
                pick = int(p.get("pick")) if p.get("pick") is not None else snake_pick(slot, rnd, field_size)
            except Exception:
                pick = snake_pick(slot, rnd, field_size)
            round_map[rnd] = {
                "name": name,
                "pos": pos,
                "round": rnd,
                "pick": pick,
                "confidence": p.get("confidence"),
            }
        by_slot[slot] = {
            "slot": slot,
            "name": str(t.get("name") or f"Team {slot}").strip(),
            "players": [round_map[r] for r in sorted(round_map)],
        }

    clean = []
    for slot in range(1, field_size+1):
        clean.append(by_slot.get(slot, {"slot": slot, "name": f"Team {slot}", "players": []}))

    # Add explicit review rows for every missing round, but the frontend only exposes
    # manual correction when the overall board passes its quality gate.
    review_keys = {(r["team_slot"], r["round"]) for r in review}
    for team in clean:
        have = {p["round"] for p in team["players"]}
        for rnd in range(1, expected_per_team+1):
            if rnd not in have and (team["slot"], rnd) not in review_keys:
                review.append({
                    "team_slot": team["slot"],
                    "round": rnd,
                    "visible_text": "",
                    "reason": "No player was confidently recognized in this roster cell.",
                })

    expected_players = field_size * expected_per_team
    found = sum(len(t["players"]) for t in clean)
    completeness = found / expected_players if expected_players else 0
    nonempty_teams = sum(1 for t in clean if len(t["players"]) >= max(3, expected_per_team//2))
    missing = expected_players - found

    quality_failed = (
        completeness < 0.90
        or missing > 12
        or nonempty_teams < field_size
        or any(len(t["players"]) > expected_per_team for t in clean)
    )
    reason_bits = []
    if completeness < 0.90:
        reason_bits.append(f"completeness {round(completeness*100)}% is below 90%")
    if missing > 12:
        reason_bits.append(f"{missing} cells are missing")
    if nonempty_teams < field_size:
        reason_bits.append(f"only {nonempty_teams}/{field_size} teams have substantial roster data")
    if quality_failed:
        warnings.append("Board read failed V25 quality gate; pod should not be accepted yet.")

    return {
        "platform_detected": data.get("platform_detected") or platform_hint,
        "teams": clean,
        "expected_players": expected_players,
        "review_cells": review,
        "warnings": warnings,
        "reader_model": MODEL,
        "quality_gate": {
            "status": "failed" if quality_failed else "passed",
            "completeness": round(completeness * 100),
            "missing_cells": missing,
            "reason": "; ".join(reason_bits) if reason_bits else "Board is complete enough for review/confirmation.",
        },
    }

@app.post("/api/vision/board")
async def vision_board(
    image: UploadFile = File(...),
    tournament_id: str = Form(...),
    platform: str = Form(...),
    field_size: int = Form(...),
    contest_type: str = Form("advancement"),
    username: str = Form("YOU"),
):
    if not configured():
        raise HTTPException(503, "Hosted vision service is online but OPENAI_API_KEY is not configured.")
    if field_size < 2 or field_size > 12:
        raise HTTPException(400, "field_size must be between 2 and 12.")
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty screenshot.")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Screenshot is too large.")

    expected_rounds = 20 if platform.lower() == "draftkings" else 18
    views = prepare_views(raw)

    prompt = f"""
You are the Fantasy Command V25 draft-board reader.

This request contains FOUR views of the SAME board in this order:
1. FULL BOARD
2. TOP ZOOM
3. MIDDLE ZOOM
4. BOTTOM ZOOM
The zooms overlap intentionally. Use them to read tiny phone-screenshot text, but never count the same draft cell twice.

Platform hint: {platform}
Expected field size: {field_size}
Expected rounds per team: {expected_rounds}
Contest type hint: {contest_type}
User username hint: {username}

Return ONLY valid JSON, no markdown.

Schema:
{{
  "platform_detected": "Underdog or DraftKings or Unknown",
  "teams": [
    {{
      "slot": 1,
      "name": "drafter/team username exactly as visible",
      "players": [
        {{
          "name": "player name exactly as visible",
          "pos": "QB|RB|WR|TE|DST|K",
          "round": 1,
          "pick": 1,
          "confidence": 0.0
        }}
      ]
    }}
  ],
  "review_cells": [
    {{
      "team_slot": 1,
      "round": 1,
      "visible_text": "uncertain text",
      "reason": "why this needs human review"
    }}
  ],
  "warnings": []
}}

STRICT GRID RULES:
- Treat the screenshot as a {field_size}-column snake-draft board with exactly {expected_rounds} roster rounds per team.
- Return exactly {field_size} team objects, one for each slot 1 through {field_size}, in left-to-right board order.
- A team may have AT MOST one player for each round 1 through {expected_rounds}. Never return 19 or 20 players for an 18-round Underdog team.
- Read EVERY team, not only the highlighted/user column.
- Preserve the board's team/username headers as closely as possible.
- Use the full-board image to understand column and row structure; use the zoom images to decipher text.
- Follow snake order when overall pick numbers are not printed.
- If a player's NAME or POSITION is genuinely unreadable, omit that player and add exactly that slot+round to review_cells.
- Do not move a readable player to a different round just to fill a gap.
- Do not invent names, positions, teams, rounds, or picks.
- Ignore duplicate sightings created by overlapping zoom images.
- Before answering, internally verify that each team's round numbers are unique and between 1 and {expected_rounds}.
"""

    try:
        client = OpenAI()
        content = [{"type": "input_text", "text": prompt}]
        for label, url in views:
            content.append({"type": "input_text", "text": label})
            content.append({"type": "input_image", "image_url": url, "detail": "high"})
        response = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )
        data = extract_json(response.output_text)
        return validate_result(data, field_size, platform)
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        if "429" in msg or "credits" in msg.lower() or "quota" in msg.lower():
            raise HTTPException(429, "OpenAI API credits are unavailable. Add API credits, then re-read the stored screenshot.")
        raise HTTPException(500, f"Vision processing failed: {type(e).__name__}: {msg[:300]}")
