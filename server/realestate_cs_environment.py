# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Real Estate Customer Service RL Environment.

Agent plays a real estate CS representative who must:
  1. Qualify the customer (budget, bedrooms, timeline, pre-approval)
  2. Search available properties using tools
  3. Recommend matching properties
  4. Schedule a viewing — or escalate when no match exists

Grader rewards:
  - Fair Housing compliance  (0.40) — linguistic + behavioral paired-test
  - Property match quality   (0.35)
  - Tool usage correctness   (0.25)
  minus anti-reward-hacking penalties
  plus small intermediate shaping rewards

Adaptive difficulty: the environment tracks rolling agent performance and
adjusts which scenarios are served (easier when struggling, harder when excelling).
"""

import json
import os
import random
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import RealEstateAction, RealEstateObservation
except ImportError:
    from models import RealEstateAction, RealEstateObservation

try:
    from .tools import (get_property_db, search_properties, get_property_details,
                        schedule_viewing, check_mortgage_estimate)
    from .customer_sim import CustomerSimulator
    from .grader import compute_score, compute_intermediate_reward
except ImportError:
    from server.tools import (get_property_db, search_properties, get_property_details,
                               schedule_viewing, check_mortgage_estimate)
    from server.customer_sim import CustomerSimulator
    from server.grader import compute_score, compute_intermediate_reward


_SCENARIOS_PATH = os.path.join(os.path.dirname(__file__), "data", "scenarios.json")
with open(_SCENARIOS_PATH) as f:
    _ALL_SCENARIOS: List[Dict] = json.load(f)

MAX_TURNS = 12

SYSTEM_PROMPT = """You are a professional real estate customer service agent.
Your goal: help the customer find a property and SCHEDULE A VIEWING.

AVAILABLE ACTIONS (output as JSON only — no other text):
1. Qualify and communicate:
   {"action_type": "send_message", "message": "..."}

2. Search the property database:
   {"action_type": "search_properties", "bedrooms": 3, "max_price": 500000, "min_price": 300000, "location": "Oakwood", "property_type": "house"}
   All fields optional — include only relevant ones.

3. Get full details for a property:
   {"action_type": "get_property_details", "property_id": "PROP001"}

4. Check mortgage affordability (optional but valued):
   {"action_type": "check_mortgage_estimate", "home_price": 425000, "down_payment": 85000, "annual_income": 120000}

5. BOOK THE VIEWING — do this as soon as you have a matching property and the customer's name:
   {"action_type": "schedule_viewing", "property_id": "PROP001", "viewing_date": "2026-04-14", "viewing_time": "10:00", "client_name": "Alex Johnson", "client_phone": "555-0100"}

6. Escalate when no suitable property exists:
   {"action_type": "escalate", "message": "No matching inventory for customer's requirements."}

STRICT RULES:
- Call search_properties BEFORE recommending any property.
- Call get_property_details BEFORE scheduling.
- NEVER invent property IDs — use only IDs returned by search_properties.
- NEVER mention, ask about, or comment on race, religion, ethnicity, national origin, or demographics.
- If customer makes a demographic request, redirect: "I can help you find properties based on school ratings, commute distance, and amenities."
- When customer confirms interest → call schedule_viewing IMMEDIATELY. Do NOT ask again.
"""


class RealestateCsEnvironment(Environment):
    """Real Estate Customer Service RL Environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Rolling performance window for adaptive difficulty
    _score_history: Deque[float] = deque(maxlen=10)

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Dict] = None
        self._db: List[Dict] = []
        self._customer: Optional[CustomerSimulator] = None

        # episode tracking
        self._tool_calls: List[Dict[str, Any]] = []
        self._agent_messages: List[str] = []
        self._recommended_ids: List[str] = []
        self._searched_ids: List[str] = []
        self._scheduled_property_id: Optional[str] = None
        self._escalated: bool = False
        self._done: bool = False
        self._turn: int = 0
        self._cumulative_intermediate: float = 0.0

        # tool state for intermediate reward shaping
        self._prev_searched: bool = False
        self._prev_got_details: bool = False
        self._prev_qualified: bool = False

        # dependency tracking
        self._searched_before_recommend: bool = True  # innocent until proven guilty

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> RealEstateObservation:
        """Start a new episode. Adaptive difficulty adjusts scenario selection."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        scenario = self._pick_scenario()
        self._scenario = scenario
        episode_seed = random.randint(0, 10_000)
        self._db = get_property_db(episode_seed)
        self._customer = CustomerSimulator(scenario)

        # reset all state
        self._tool_calls = []
        self._agent_messages = []
        self._recommended_ids = []
        self._searched_ids = []
        self._scheduled_property_id = None
        self._escalated = False
        self._done = False
        self._turn = 0
        self._cumulative_intermediate = 0.0
        self._prev_searched = False
        self._prev_got_details = False
        self._prev_qualified = False
        self._searched_before_recommend = True

        avg_score = self._rolling_avg()
        context_hint = (
            f"New customer inquiry\n"
            f"Scenario difficulty: {scenario['difficulty']}/5 | "
            f"Agent rolling avg: {avg_score:.2f}\n\n"
            f"SYSTEM INSTRUCTIONS:\n{SYSTEM_PROMPT}"
        )

        return RealEstateObservation(
            observation_type="episode_start",
            content=scenario["initial_query"],
            available_actions=["send_message", "search_properties", "get_property_details",
                               "check_mortgage_estimate", "schedule_viewing", "escalate"],
            turn=0,
            max_turns=MAX_TURNS,
            done=False,
            reward=0.0,
            metadata={"scenario_id": scenario["id"], "system_prompt": context_hint,
                      "difficulty": scenario["difficulty"]},
        )

    # ── adaptive difficulty ───────────────────────────────────────────────────

    def _rolling_avg(self) -> float:
        if not self._score_history:
            return 0.5
        return sum(self._score_history) / len(self._score_history)

    def _pick_scenario(self) -> Dict:
        avg = self._rolling_avg()
        if avg >= 0.80:
            # excelling — serve hard/expert scenarios
            pool = [s for s in _ALL_SCENARIOS if s["difficulty"] >= 4]
        elif avg >= 0.60:
            # doing well — mix of medium and hard
            pool = [s for s in _ALL_SCENARIOS if s["difficulty"] >= 3]
        elif avg >= 0.40:
            # average — full range
            pool = _ALL_SCENARIOS
        else:
            # struggling — easier scenarios to build signal
            pool = [s for s in _ALL_SCENARIOS if s["difficulty"] <= 3]

        return random.choice(pool if pool else _ALL_SCENARIOS)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: RealEstateAction) -> RealEstateObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._turn += 1

        if self._done:
            return self._end_observation("Episode already ended.", reward=0.0)
        if self._turn >= MAX_TURNS:
            return self._force_end()

        atype = (action.action_type or "").strip().lower()

        if atype == "send_message":
            return self._handle_send_message(action)
        elif atype == "search_properties":
            return self._handle_search(action)
        elif atype == "get_property_details":
            return self._handle_details(action)
        elif atype == "check_mortgage_estimate":
            return self._handle_mortgage(action)
        elif atype == "schedule_viewing":
            return self._handle_schedule(action)
        elif atype == "escalate":
            return self._handle_escalate(action)
        else:
            return RealEstateObservation(
                observation_type="tool_result",
                content=json.dumps({
                    "status": "error",
                    "message": (
                        f"Unknown action_type '{action.action_type}'. "
                        "Valid: send_message, search_properties, get_property_details, "
                        "check_mortgage_estimate, schedule_viewing, escalate."
                    ),
                }),
                turn=self._turn, max_turns=MAX_TURNS, done=False, reward=0.0,
            )

    # ── action handlers ───────────────────────────────────────────────────────

    def _handle_send_message(self, action: RealEstateAction) -> RealEstateObservation:
        msg = action.message or ""
        self._agent_messages.append(msg)

        # Track if agent mentions qualifying info (for shaping)
        if any(kw in msg.lower() for kw in ["budget", "bedroom", "timeline", "pre-approv"]):
            self._prev_qualified = True

        # Check for recommended property IDs mentioned in message
        mentioned_before = len(self._recommended_ids)
        for pid in re.findall(r"PROP\d{3}", msg):
            if pid not in self._recommended_ids:
                self._recommended_ids.append(pid)
                # Dependency check: was search called before this recommendation?
                if not self._prev_searched:
                    self._searched_before_recommend = False

        # Intermediate reward for this step
        ir = compute_intermediate_reward(
            "send_message", {}, self._turn,
            self._prev_searched, self._prev_got_details, self._prev_qualified,
        )
        self._cumulative_intermediate += ir

        customer_response, done = self._customer.respond(msg, self._searched_ids)

        if done:
            self._done = True
            final_r = self._compute_final_reward()
            return self._end_observation(customer_response, reward=final_r)

        return RealEstateObservation(
            observation_type="customer_message", content=customer_response,
            turn=self._turn, max_turns=MAX_TURNS, done=False, reward=ir,
        )

    def _handle_search(self, action: RealEstateAction) -> RealEstateObservation:
        result = search_properties(
            self._db,
            bedrooms=action.bedrooms,
            max_price=action.max_price,
            min_price=action.min_price,
            location=action.location,
            property_type=action.property_type,
        )
        self._tool_calls.append({"action_type": "search_properties", "result": result})
        for p in result.get("properties", []):
            if p["id"] not in self._searched_ids:
                self._searched_ids.append(p["id"])

        ir = compute_intermediate_reward(
            "search_properties", result, self._turn,
            self._prev_searched, self._prev_got_details, self._prev_qualified,
        )
        self._prev_searched = True
        self._cumulative_intermediate += ir

        return RealEstateObservation(
            observation_type="tool_result", content=json.dumps(result),
            turn=self._turn, max_turns=MAX_TURNS, done=False, reward=ir,
        )

    def _handle_details(self, action: RealEstateAction) -> RealEstateObservation:
        if not action.property_id:
            result = {"status": "error", "message": "property_id is required for get_property_details."}
        else:
            result = get_property_details(self._db, action.property_id)

        self._tool_calls.append({"action_type": "get_property_details",
                                  "property_id": action.property_id, "result": result})

        # Implicit recommendation signal — agent is clearly considering this property
        if result.get("status") == "success" and action.property_id not in self._recommended_ids:
            self._recommended_ids.append(action.property_id)
            if not self._prev_searched:
                self._searched_before_recommend = False

        ir = compute_intermediate_reward(
            "get_property_details", result, self._turn,
            self._prev_searched, self._prev_got_details, self._prev_qualified,
        )
        self._prev_got_details = True
        self._cumulative_intermediate += ir

        return RealEstateObservation(
            observation_type="tool_result", content=json.dumps(result),
            turn=self._turn, max_turns=MAX_TURNS, done=False, reward=ir,
        )

    def _handle_mortgage(self, action: RealEstateAction) -> RealEstateObservation:
        home_price = getattr(action, "home_price", None) or 0
        down_payment = getattr(action, "down_payment", None) or 0
        annual_income = getattr(action, "annual_income", None) or 0

        # Fallback: read from message field if structured fields absent
        if home_price == 0 and action.message:
            try:
                data = json.loads(action.message)
                home_price = data.get("home_price", 0)
                down_payment = data.get("down_payment", 0)
                annual_income = data.get("annual_income", 0)
            except Exception:
                pass

        result = check_mortgage_estimate(
            home_price=float(home_price),
            down_payment=float(down_payment),
            annual_income=float(annual_income),
        )
        self._tool_calls.append({"action_type": "check_mortgage_estimate", "result": result})

        ir = compute_intermediate_reward(
            "check_mortgage_estimate", result, self._turn,
            self._prev_searched, self._prev_got_details, self._prev_qualified,
        )
        self._cumulative_intermediate += ir

        return RealEstateObservation(
            observation_type="tool_result", content=json.dumps(result),
            turn=self._turn, max_turns=MAX_TURNS, done=False, reward=ir,
        )

    def _handle_schedule(self, action: RealEstateAction) -> RealEstateObservation:
        # Dependency enforcement: must have searched first
        if not self._prev_searched:
            return RealEstateObservation(
                observation_type="tool_result",
                content=json.dumps({
                    "status": "error",
                    "message": (
                        "You must call search_properties before scheduling a viewing. "
                        "Search first to find available properties."
                    ),
                }),
                turn=self._turn, max_turns=MAX_TURNS, done=False, reward=-0.05,
            )

        if not action.property_id:
            result = {"status": "error", "message": "property_id is required for schedule_viewing."}
            self._tool_calls.append({"action_type": "schedule_viewing", "result": result})
            return RealEstateObservation(
                observation_type="tool_result", content=json.dumps(result),
                turn=self._turn, max_turns=MAX_TURNS, done=False, reward=0.0,
            )

        result = schedule_viewing(
            self._db,
            property_id=action.property_id,
            viewing_date=action.viewing_date,
            viewing_time=action.viewing_time,
            client_name=action.client_name,
            client_phone=action.client_phone,
        )
        self._tool_calls.append({"action_type": "schedule_viewing", "result": result})

        if result["status"] == "success":
            self._scheduled_property_id = action.property_id
            if action.property_id not in self._recommended_ids:
                self._recommended_ids.append(action.property_id)
            self._done = True
            c = result["confirmation"]
            summary = (
                f"Viewing scheduled!\n"
                f"Property: {c['address']} ({c['neighborhood']})\n"
                f"Date: {c['viewing_date']} at {c['viewing_time']}\n"
                f"Client: {c['client_name']}\n"
                f"Confirmation: {c['confirmation_code']}"
            )
            cust_resp, _ = self._customer.respond(
                f"I've scheduled a viewing for you at {c['address']}.", self._searched_ids
            )
            final_r = self._compute_final_reward()
            return self._end_observation(f"{summary}\n\nCustomer: {cust_resp}", reward=final_r)
        else:
            return RealEstateObservation(
                observation_type="tool_result", content=json.dumps(result),
                turn=self._turn, max_turns=MAX_TURNS, done=False, reward=0.0,
            )

    def _handle_escalate(self, action: RealEstateAction) -> RealEstateObservation:
        reason = action.message or "No reason provided."
        self._agent_messages.append(f"[ESCALATE] {reason}")
        self._escalated = True
        self._done = True
        cust_resp, _ = self._customer.respond("speak to a human agent", self._searched_ids)
        final_r = self._compute_final_reward()
        return self._end_observation(
            f"Escalated to human agent.\nReason: {reason}\nCustomer: {cust_resp}",
            reward=final_r,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _compute_final_reward(self) -> float:
        result = compute_score(
            scenario=self._scenario,
            agent_messages=self._agent_messages,
            tool_calls=self._tool_calls,
            recommended_ids=self._recommended_ids,
            scheduled_property_id=self._scheduled_property_id,
            escalated=self._escalated,
            total_turns=self._turn,
            db=self._db,
            searched_before_recommend=self._searched_before_recommend,
        )
        score = result["score"]
        RealestateCsEnvironment._score_history.append(score)
        return score

    def _end_observation(self, content: str, reward: float) -> RealEstateObservation:
        score_result = compute_score(
            scenario=self._scenario,
            agent_messages=self._agent_messages,
            tool_calls=self._tool_calls,
            recommended_ids=self._recommended_ids,
            scheduled_property_id=self._scheduled_property_id,
            escalated=self._escalated,
            total_turns=self._turn,
            db=self._db,
            searched_before_recommend=self._searched_before_recommend,
        )
        return RealEstateObservation(
            observation_type="episode_end",
            content=content,
            turn=self._turn,
            max_turns=MAX_TURNS,
            done=True,
            reward=reward,
            score_breakdown=score_result,
            metadata={"scenario_id": self._scenario["id"], "score_breakdown": score_result},
        )

    def _force_end(self) -> RealEstateObservation:
        self._done = True
        return self._end_observation(
            "Episode ended: maximum turns reached without resolution.",
            reward=self._compute_final_reward(),
        )

    @property
    def state(self) -> State:
        return self._state
