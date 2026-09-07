import os, base64, json, re
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="Fantasy Command Vision Service", version="23")

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

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "Fantasy Command V23 Vision Service",
        "version": "V23",
        "vision_configured": configured(),
        "model": MODEL if configured() else None,
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

def validate_result(data: dict, field_size: int, platform_hint: str) -> dict:
    teams = data.get("teams") or data.get("pod") or []
    if not isinstance(teams, list) or not teams:
        raise HTTPException(422, "Vision model returned no teams.")
    clean=[]
    review=list(data.get("review_cells") or [])
    warnings=list(data.get("warnings") or [])
    for i,t in enumerate(teams[:12]):
        players=[]
        for j,p in enumerate((t.get("players") or [])):
            name=str(p.get("name") or "").strip()
            pos=str(p.get("pos") or p.get("position") or "").upper().strip()
            if pos in ("DST","DEF","D/ST"): pos="DST"
            if name and pos in ("QB","RB","WR","TE","DST","K"):
                players.append({
                    "name":name,
                    "pos":pos,
                    "round":int(p.get("round") or j+1),
                    "pick":p.get("pick"),
                    "confidence":p.get("confidence"),
                })
        clean.append({
            "slot":int(t.get("slot") or i+1),
            "name":str(t.get("name") or f"Team {i+1}").strip(),
            "players":players,
        })
    if len(clean) != field_size:
        warnings.append(f"Expected {field_size} teams; reader returned {len(clean)}.")
    expected_per_team = 20 if platform_hint.lower()=="draftkings" else 18
    expected_players = field_size * expected_per_team
    found=sum(len(t["players"]) for t in clean)
    if found < expected_players:
        warnings.append(f"{expected_players-found} roster cells were not confidently recognized.")
    return {
        "platform_detected": data.get("platform_detected") or platform_hint,
        "teams":clean,
        "expected_players":expected_players,
        "review_cells":review,
        "warnings":warnings,
        "reader_model":MODEL,
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
    mime=image.content_type or "image/png"
    data_url=f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    prompt=f"""
You are the draft-board reader for Fantasy Command.

Read this FULL fantasy-football draft-board screenshot with extreme care.
Platform hint: {platform}
Expected field size: {field_size}
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

Rules:
- Preserve board column order as team slots.
- Follow snake-draft order when assigning overall pick numbers if overall picks are not printed.
- Do not invent a player if the cell is unreadable. Put that cell in review_cells.
- Use the position shown on the board. If position is not legible, do not guess.
- Read all visible rounds, not only the user's team.
- Underdog standard boards usually have 18 rounds; DraftKings Best Ball commonly has 20. Use the screenshot itself as the authority.
- The expected number of teams is {field_size}; do not silently add teams.
- Team/drafter names matter. Read them from the column headers.
- confidence is from 0 to 1 and reflects visual confidence in that cell.
"""
    try:
        client=OpenAI()
        response=client.responses.create(
            model=MODEL,
            input=[{
                "role":"user",
                "content":[
                    {"type":"input_text","text":prompt},
                    {"type":"input_image","image_url":data_url,"detail":"high"},
                ],
            }],
        )
        data=extract_json(response.output_text)
        return validate_result(data, field_size, platform)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Vision processing failed: {type(e).__name__}: {str(e)[:300]}")
