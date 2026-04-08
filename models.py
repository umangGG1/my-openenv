# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Real Estate Customer Service RL Environment.

The agent takes typed actions (tool calls or messages) and receives
structured observations (customer messages or tool results).
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class RealEstateAction(Action):
    """
    An action the agent can take in the real estate CS environment.

    action_type must be one of:
      - send_message        : send a text message to the customer
      - search_properties   : search the MLS database
      - get_property_details: get full details for a specific property
      - schedule_viewing    : book a property viewing (terminal action)
      - escalate            : hand off to a human agent (terminal action)
    """

    action_type: str = Field(
        ...,
        description=(
            "Action type. One of: send_message, search_properties, "
            "get_property_details, schedule_viewing, escalate"
        ),
    )

    # send_message / escalate
    message: Optional[str] = Field(
        None,
        description="Text to send to the customer (for send_message / escalate).",
    )

    # search_properties filters
    bedrooms: Optional[int] = Field(None, description="Required number of bedrooms.")
    max_price: Optional[float] = Field(None, description="Maximum price in USD.")
    min_price: Optional[float] = Field(None, description="Minimum price in USD.")
    location: Optional[str] = Field(None, description="Neighborhood name to filter by.")
    property_type: Optional[str] = Field(
        None, description="Property type filter: house, condo, or apartment."
    )

    # get_property_details / schedule_viewing
    property_id: Optional[str] = Field(
        None,
        description="Property ID (e.g. PROP001). Required for get_property_details and schedule_viewing.",
    )

    # schedule_viewing extras
    viewing_date: Optional[str] = Field(
        None, description="Preferred viewing date in YYYY-MM-DD format."
    )
    viewing_time: Optional[str] = Field(
        None, description="Preferred viewing time in HH:MM format."
    )
    client_name: Optional[str] = Field(
        None, description="Client's full name (required for schedule_viewing)."
    )
    client_phone: Optional[str] = Field(
        None, description="Client's phone number (optional for schedule_viewing)."
    )

    # check_mortgage_estimate fields
    home_price: Optional[float] = Field(None, description="Home purchase price in USD.")
    down_payment: Optional[float] = Field(None, description="Down payment amount in USD.")
    annual_income: Optional[float] = Field(None, description="Gross annual income in USD.")
    interest_rate: Optional[float] = Field(None, description="Annual interest rate % (default 7.0).")
    loan_term_years: Optional[int] = Field(None, description="Loan term in years (default 30).")


class RealEstateObservation(Observation):
    """
    An observation returned to the agent after each action.

    observation_type indicates the source:
      - episode_start   : initial customer query + scenario context
      - customer_message: customer replied to a send_message
      - tool_result     : JSON result from a tool call
      - episode_end     : episode over, final reward available in `reward`
    """

    observation_type: str = Field(
        default="customer_message",
        description="Type: episode_start | customer_message | tool_result | episode_end",
    )
    content: str = Field(
        default="",
        description="Customer message, tool result JSON, or episode summary.",
    )
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "send_message",
            "search_properties",
            "get_property_details",
            "check_mortgage_estimate",
            "schedule_viewing",
            "escalate",
        ],
        description="Actions currently available to the agent.",
    )
    turn: int = Field(default=0, description="Current turn number (starts at 0).")
    max_turns: int = Field(default=12, description="Maximum turns per episode.")
    score_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Grader score breakdown (only present in episode_end observations).",
    )
