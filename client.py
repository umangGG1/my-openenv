# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real Estate CS Environment Client."""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import RealEstateAction, RealEstateObservation


class RealEstateCsEnv(EnvClient[RealEstateAction, RealEstateObservation, State]):
    """
    Client for the Real Estate Customer Service Environment.

    Connect to a running server and interact via reset() / step().

    Example::

        with RealEstateCsEnv(base_url="http://localhost:8000") as env:
            result = env.reset()
            print(result.observation.content)   # initial customer query

            result = env.step(RealEstateAction(
                action_type="search_properties",
                bedrooms=3,
                max_price=480000,
            ))
            print(result.observation.content)   # JSON search results
    """

    def _step_payload(self, action: RealEstateAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.message is not None:
            payload["message"] = action.message
        if action.bedrooms is not None:
            payload["bedrooms"] = action.bedrooms
        if action.max_price is not None:
            payload["max_price"] = action.max_price
        if action.min_price is not None:
            payload["min_price"] = action.min_price
        if action.location is not None:
            payload["location"] = action.location
        if action.property_type is not None:
            payload["property_type"] = action.property_type
        if action.property_id is not None:
            payload["property_id"] = action.property_id
        if action.viewing_date is not None:
            payload["viewing_date"] = action.viewing_date
        if action.viewing_time is not None:
            payload["viewing_time"] = action.viewing_time
        if action.client_name is not None:
            payload["client_name"] = action.client_name
        if action.client_phone is not None:
            payload["client_phone"] = action.client_phone
        if action.home_price is not None:
            payload["home_price"] = action.home_price
        if action.down_payment is not None:
            payload["down_payment"] = action.down_payment
        if action.annual_income is not None:
            payload["annual_income"] = action.annual_income
        if action.interest_rate is not None:
            payload["interest_rate"] = action.interest_rate
        if action.loan_term_years is not None:
            payload["loan_term_years"] = action.loan_term_years
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[RealEstateObservation]:
        obs_data = payload.get("observation", {})
        observation = RealEstateObservation(
            observation_type=obs_data.get("observation_type", "customer_message"),
            content=obs_data.get("content", ""),
            available_actions=obs_data.get("available_actions", []),
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 12),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            score_breakdown=obs_data.get("score_breakdown"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
