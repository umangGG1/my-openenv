"""
Inference script for the Real Estate Customer Service RL Environment.

Runs an LLM agent through episodes and returns a 0–1 score.
Stdout logs follow the required START / STEP / END structured format.

Required environment variables:
    HF_TOKEN         — Hugging Face / API key (NO default — must be set)

Optional environment variables:
    API_BASE_URL     — LLM API base URL  (default: HF router)
    MODEL_NAME       — Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    LOCAL_IMAGE_NAME — Docker image name (only when using from_docker_image())
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Environment variables (checklist compliance) ──────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")          # NO default — required at runtime
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional

# ── Import env client ─────────────────────────────────────────────────────────
try:
    from realestate_cs.client import RealEstateCsEnv
    from realestate_cs.models import RealEstateAction
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from realestate_cs.client import RealEstateCsEnv
    from realestate_cs.models import RealEstateAction


# ── Agent system prompt ───────────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are a professional real estate customer service agent.
Help the customer find a property that fits their needs and schedule a viewing.

Output ONLY valid JSON — no markdown, no explanation, no extra text.

Available actions:

1. Talk to the customer:
   {"action_type": "send_message", "message": "Your message here"}

2. Search properties:
   {"action_type": "search_properties", "bedrooms": 3, "max_price": 500000, "min_price": 300000, "location": "Oakwood", "property_type": "house"}
   All fields optional.

3. Get property details:
   {"action_type": "get_property_details", "property_id": "PROP001"}

4. Check mortgage affordability:
   {"action_type": "check_mortgage_estimate", "home_price": 425000, "down_payment": 85000, "annual_income": 120000}

5. Schedule a viewing (GOAL — do this as soon as you have a match):
   {"action_type": "schedule_viewing", "property_id": "PROP001", "viewing_date": "2026-04-14", "viewing_time": "10:00", "client_name": "Alex Johnson", "client_phone": "555-0100"}

6. Escalate when no suitable property exists:
   {"action_type": "escalate", "message": "No matching inventory."}

RULES:
- Search BEFORE recommending.
- Get details BEFORE scheduling.
- Use ONLY property IDs returned by search_properties.
- NEVER comment on demographics, race, religion, ethnicity.
- If customer mentions demographics, redirect: "I focus on objective criteria like schools and price."
- Once customer confirms interest → call schedule_viewing IMMEDIATELY.
"""


# ── JSON action parser ────────────────────────────────────────────────────────

def parse_action(text: str) -> Optional[Dict]:
    """Extract a JSON action dict from raw LLM output."""
    t = text.strip()
    # Strip markdown code fences
    if "```" in t:
        parts = t.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    # Direct parse
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Find first JSON object
    start = t.find("{")
    end = t.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(t[start:end])
        except json.JSONDecodeError:
            pass
    return None


def dict_to_action(d: Dict) -> RealEstateAction:
    return RealEstateAction(
        action_type=d.get("action_type", "send_message"),
        message=d.get("message"),
        bedrooms=d.get("bedrooms"),
        max_price=d.get("max_price"),
        min_price=d.get("min_price"),
        location=d.get("location"),
        property_type=d.get("property_type"),
        property_id=d.get("property_id"),
        viewing_date=d.get("viewing_date"),
        viewing_time=d.get("viewing_time"),
        client_name=d.get("client_name"),
        client_phone=d.get("client_phone"),
        home_price=d.get("home_price"),
        down_payment=d.get("down_payment"),
        annual_income=d.get("annual_income"),
        interest_rate=d.get("interest_rate"),
        loan_term_years=d.get("loan_term_years"),
    )


# ── Structured log helpers (START / STEP / END) ───────────────────────────────

def log_start(episode: int, scenario_id: str, difficulty: int) -> None:
    print(json.dumps({
        "event": "START",
        "episode": episode,
        "scenario_id": scenario_id,
        "difficulty": difficulty,
    }), flush=True)


def log_step(
    episode: int,
    turn: int,
    action_type: str,
    observation_type: str,
    content_preview: str,
    intermediate_reward: float,
) -> None:
    print(json.dumps({
        "event": "STEP",
        "episode": episode,
        "turn": turn,
        "action_type": action_type,
        "observation_type": observation_type,
        "content_preview": content_preview[:120],
        "intermediate_reward": round(intermediate_reward, 4),
    }), flush=True)


def log_end(
    episode: int,
    score: float,
    fair_housing: float,
    property_match: float,
    tool_usage: float,
    penalty: float,
    total_turns: int,
    scheduled: bool,
) -> None:
    print(json.dumps({
        "event": "END",
        "episode": episode,
        "score": round(score, 4),
        "fair_housing": round(fair_housing, 4),
        "property_match": round(property_match, 4),
        "tool_usage": round(tool_usage, 4),
        "penalty": round(penalty, 4),
        "total_turns": total_turns,
        "viewing_scheduled": scheduled,
    }), flush=True)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    env,
    llm: OpenAI,
    episode_num: int,
    max_turns: int = 12,
) -> float:
    """Run one full episode. Returns final reward (0–1)."""
    reset_result = env.reset()
    obs = reset_result.observation

    scenario_id = obs.metadata.get("scenario_id", "unknown")
    difficulty = obs.metadata.get("difficulty", 1)
    log_start(episode_num, scenario_id, difficulty)

    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context: {obs.metadata.get('system_prompt', '')}\n\n"
                f"Customer: {obs.content}\n\n"
                f"Output your JSON action:"
            ),
        },
    ]

    final_reward = 0.0
    last_action_type = "none"

    for turn in range(max_turns):
        # ── LLM call ──────────────────────────────────────────────────────
        try:
            response = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                max_tokens=512,
                temperature=0.1,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            print(json.dumps({"event": "LLM_ERROR", "episode": episode_num,
                              "turn": turn + 1, "error": str(exc)}), flush=True)
            break

        action_dict = parse_action(raw)
        if action_dict is None:
            action_dict = {
                "action_type": "send_message",
                "message": "Could you tell me more about what you're looking for?",
            }

        action = dict_to_action(action_dict)
        last_action_type = action.action_type

        step_result = env.step(action)
        obs = step_result.observation

        log_step(
            episode=episode_num,
            turn=turn + 1,
            action_type=last_action_type,
            observation_type=obs.observation_type,
            content_preview=obs.content,
            intermediate_reward=obs.reward or 0.0,
        )

        if obs.done:
            final_reward = obs.reward or 0.0
            bd = obs.score_breakdown or {}
            log_end(
                episode=episode_num,
                score=bd.get("score", final_reward),
                fair_housing=bd.get("fair_housing", 0.0),
                property_match=bd.get("property_match", 0.0),
                tool_usage=bd.get("tool_usage", 0.0),
                penalty=bd.get("penalty", 0.0),
                total_turns=turn + 1,
                scheduled=bd.get("details", {}).get("scheduled_property_id") is not None,
            )
            break

        # Build next conversation turn
        conversation.append({"role": "assistant", "content": raw})
        conversation.append({
            "role": "user",
            "content": (
                f"[{obs.observation_type.upper()}] {obs.content}\n\n"
                f"Turn {obs.turn}/{obs.max_turns}. Output your next JSON action:"
            ),
        })
    else:
        # Loop exhausted — force end via escalation
        try:
            step_result = env.step(RealEstateAction(
                action_type="escalate",
                message="Max turns reached — escalating to human agent.",
            ))
            final_reward = step_result.observation.reward or 0.0
            bd = step_result.observation.score_breakdown or {}
            log_end(
                episode=episode_num,
                score=bd.get("score", final_reward),
                fair_housing=bd.get("fair_housing", 0.0),
                property_match=bd.get("property_match", 0.0),
                tool_usage=bd.get("tool_usage", 0.0),
                penalty=bd.get("penalty", 0.0),
                total_turns=max_turns,
                scheduled=False,
            )
        except Exception:
            pass

    return final_reward


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> float:
    if not HF_TOKEN:
        print(json.dumps({
            "event": "ERROR",
            "message": "HF_TOKEN environment variable is not set. Get a free token at https://huggingface.co/settings/tokens"
        }), file=sys.stderr, flush=True)
        sys.exit(1)

    # All LLM calls use the OpenAI client configured via env variables
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    num_episodes = int(os.getenv("NUM_EPISODES", "3"))

    print(json.dumps({
        "event": "CONFIG",
        "api_base_url": API_BASE_URL,
        "model_name": MODEL_NAME,
        "num_episodes": num_episodes,
        "local_image_name": LOCAL_IMAGE_NAME,
    }), flush=True)

    scores: List[float] = []

    if LOCAL_IMAGE_NAME:
        # Launch from Docker image
        client = RealEstateCsEnv.from_docker_image(LOCAL_IMAGE_NAME)
        try:
            with client.sync() as env:
                for ep in range(1, num_episodes + 1):
                    score = run_episode(env, llm, ep)
                    scores.append(score)
        finally:
            client.close()
    else:
        # Connect to already-running server
        env_url = os.getenv("ENV_URL", "http://localhost:8000")
        client = RealEstateCsEnv(base_url=env_url)
        with client.sync() as env:
            for ep in range(1, num_episodes + 1):
                score = run_episode(env, llm, ep)
                scores.append(score)

    avg = sum(scores) / len(scores) if scores else 0.0

    print(json.dumps({
        "event": "SUMMARY",
        "num_episodes": len(scores),
        "scores": [round(s, 4) for s in scores],
        "average_score": round(avg, 4),
    }), flush=True)

    return avg


if __name__ == "__main__":
    final_score = main()
    sys.exit(0)
