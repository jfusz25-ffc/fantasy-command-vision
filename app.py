import os, base64, json, re, io
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image

app = FastAPI(title="Fantasy Command Vision Service", version="28.9")

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
    return {"ok": True, "service": "Fantasy Command V28.9 Vision Service", "health": "/api/health"}

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "Fantasy Command V28.9 Vision Service",
        "version": "V28.9",
        "vision_configured": configured(),
        "model": MODEL if configured() else None,
        "phone_board_reader": True,
        "column_slice_reader": True,
        "cell_reader": True,
        "board_reader": True,
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

def prepare_views(raw: bytes, field_size: int = 12) -> list[tuple[str,str]]:
    """V28.7: full board + six two-column vertical slices.

    Phone screenshots make player text tiny because all 12 columns share a narrow
    image.  Cropping by COLUMNS (rather than only by row bands) enlarges every
    roster cell while preserving all 18/20 rounds for those teams.
    """
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:
        raise HTTPException(400, f"Could not decode screenshot: {type(e).__name__}")
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size
    # Preserve source pixels. Only shrink unusually huge screenshots.
    max_w = 2600
    if w > max_w:
        ratio = max_w / w
        im = im.resize((max_w, max(1, int(h * ratio))), Image.LANCZOS)
        w, h = im.size

    views = [("FULL BOARD — use for headers, row alignment, and global structure", image_data_url(im, 95))]

    # Six slices, two team columns each. A small overlap protects text near cell edges.
    # The model is told the exact slot range for each image so it does not have to
    # infer which board columns a crop represents.
    groups = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)] if field_size == 12 else []
    if not groups:
        step = 2
        groups = [(a, min(field_size, a+step-1)) for a in range(1, field_size+1, step)]
    col_w = w / field_size
    pad = max(2, int(col_w * 0.10))
    for a,b in groups:
        x0 = max(0, int((a-1)*col_w) - pad)
        x1 = min(w, int(b*col_w) + pad)
        crop = im.crop((x0, 0, x1, h))
        target_w = 1800
        if crop.width < target_w:
            ratio = target_w / crop.width
            crop = crop.resize((target_w, max(1, int(crop.height*ratio))), Image.LANCZOS)
        views.append((f"TEAM SLOTS {a}-{b} — full height, all rounds", image_data_url(crop, 95)))
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
    views = prepare_views(raw, field_size)

    prompt = f"""
You are the Fantasy Command V28.9 draft-board reader. Accuracy and GRID COMPLETENESS are more important than speed.

The first image is the FULL BOARD. Every later image is a full-height enlarged crop whose label tells you the exact TEAM SLOT range it contains. Use the full board for global row/header alignment and the slot crops to read player text.

Platform hint: {platform}
Expected field size: {field_size}
Expected rounds per team: {expected_rounds}
Contest type hint: {contest_type}
User username hint: {username}

Return ONLY valid JSON, no markdown, using this schema:
{{
  "platform_detected": "Underdog or DraftKings or Unknown",
  "teams": [{{"slot":1,"name":"team header","players":[{{"name":"player","pos":"QB|RB|WR|TE|DST|K","round":1,"pick":1,"confidence":0.0}}]}}],
  "review_cells": [{{"team_slot":1,"round":1,"visible_text":"uncertain text","reason":"why review is needed"}}],
  "warnings": []
}}

MANDATORY GRID PROCEDURE:
1. Build an internal {field_size} x {expected_rounds} grid before producing JSON.
2. Process the enlarged slot crops one at a time from left to right. Each crop label gives its absolute team slots.
3. For EACH team slot, walk rounds 1 through {expected_rounds} from top to bottom. Do not skip a readable cell.
4. Return exactly {field_size} team objects, slots 1..{field_size}, in left-to-right order.
5. Each slot may contain at most one player per round. Round numbers must be unique and within 1..{expected_rounds}.
6. Read every team, not only the highlighted/user team.
7. Use the full-board image to resolve team headers and row alignment; use crops for tiny player names and positions.
8. Follow snake order for overall pick numbers when they are not printed.
9. If a cell is genuinely unreadable after checking both crop and full board, OMIT only that player and add exactly that slot+round to review_cells.
10. Never invent a player to fill a gap. Never substitute a different real NFL player merely because the text looks similar.
11. Before answering, internally count expected cells ({field_size*expected_rounds}) and verify every missing cell has a review_cells entry.
12. Preserve visible spellings as closely as possible; downstream Player Identity will handle likely spelling corrections.
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

# V28.9 targeted single-cell reader retained alongside the full-board reader.
@app.post('/api/vision/cell')
async def vision_cell(image:UploadFile=File(...),tournament_id:str=Form(...),platform:str=Form(...),field_size:int=Form(...),team_slot:int=Form(...),round:int=Form(...),pick:int=Form(...),current_read:str=Form('')):
 if not configured(): raise HTTPException(503,'Hosted vision service is online but OPENAI_API_KEY is not configured.')
 raw=await image.read()
 if not raw: raise HTTPException(400,'Empty cell image.')
 if len(raw)>5*1024*1024: raise HTTPException(413,'Cell image is too large.')
 try:
  im=Image.open(io.BytesIO(raw));im.load()
  if im.mode not in ('RGB','L'): im=im.convert('RGB')
  out=io.BytesIO();im.save(out,format='JPEG',quality=96,optimize=True);raw=out.getvalue()
 except Exception as e: raise HTTPException(400,f'Could not decode cell image: {type(e).__name__}')
 prompt=f'''You are Fantasy Command V28.8 CELL READER. The attached image is one enlarged fantasy-football draft-board cell only.
Platform: {platform}. Team slot: {team_slot}. Round: {round}. Overall pick: {pick}. Previous OCR read: {current_read or 'none'}.
Read ONLY the player name and displayed fantasy position in this single cell. Do not infer from draft rankings, ADP, the previous OCR read, or nearby expected players. Preserve what is visibly printed. If unreadable, return an empty name and explain in visible_text.
Return ONLY JSON: {{"name":"player name or empty","pos":"QB|RB|WR|TE|DST|K or empty","confidence":0.0,"visible_text":"brief literal text you can see"}}'''
 try:
  client=OpenAI();resp=client.responses.create(model=MODEL,input=[{'role':'user','content':[{'type':'input_text','text':prompt},{'type':'input_image','image_url':data_url(raw),'detail':'high'}]}])
  d=extract_json(resp.output_text);name=str(d.get('name') or '').strip();pos=str(d.get('pos') or d.get('position') or '').upper().strip();
  if pos in ('DEF','D/ST'):pos='DST'
  if pos not in ('QB','RB','WR','TE','DST','K'):pos=''
  return {'name':name,'pos':pos,'confidence':d.get('confidence'),'visible_text':str(d.get('visible_text') or ''),'team_slot':team_slot,'round':round,'pick':pick,'reader_model':MODEL}
 except HTTPException: raise
 except Exception as e:
  msg=str(e)
  if '429' in msg or 'credits' in msg.lower() or 'quota' in msg.lower(): raise HTTPException(429,'OpenAI API credits are unavailable.')
  raise HTTPException(500,f'Cell vision processing failed: {type(e).__name__}: {msg[:300]}')
