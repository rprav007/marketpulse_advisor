"""Test runner script for MarketPulse Advisor multi-agent scenarios.

This script executes happy path and circuit breaker scenarios to verify that the
Technical Scanner, Backtester, and Risk Advisor agents orchestrate correctly.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Configure environment variables to mock mode and Vertex settings
os.environ["GOOGLE_CLOUD_PROJECT"] = "rad-alm-test"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["MOCK_MODE"] = "True"

# Add current workspace to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from marketpulse_advisor.agent import root_agent


async def run_scenario(scenario_name: str, user_id: str, query: str):
    """Executes a trading query scenario and streams the multi-agent events."""
    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario_name}")
    print(f"User ID: {user_id} | Query: '{query}'")
    print("=" * 80)

    session_service = InMemorySessionService()
    session_id = f"session_{user_id}_{int(asyncio.get_event_loop().time())}"

    # Initialize session
    await session_service.create_session(
        app_name="marketpulse_advisor",
        user_id=user_id,
        session_id=session_id
    )

    # Initialize runner
    runner = Runner(
        agent=root_agent,
        app_name="marketpulse_advisor",
        session_service=session_service
    )

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=query)]
            ),
        ):
            author = event.author
            parts = event.content.parts if event.content else []
            content_text = parts[0].text if parts else ""

            if event.is_final_response():
                print(f"\n>>> [FINAL RESPONSE from {author}]:\n{content_text}\n")
            elif author == "system":
                # Check for tool call prints
                if event.content and event.content.parts:
                    print(f" [System action]: {content_text}")
            else:
                if content_text:
                    print(f" [{author}]: {content_text}")
    except Exception as e:
        print(f"❌ Scenario execution failed: {e}", file=sys.stderr)


async def main():
    print("Initializing MarketPulse Advisor Verification Runner...")
    
    # Scenario 1: Happy Path
    await run_scenario(
        scenario_name="1. Happy Path - Standard Pre-Market Recommendation",
        user_id="user_happy",
        query="Perform the pre-market scan and generate risk-sized bracket recommendations for user_happy"
    )

    # Scenario 2: Circuit Breaker - Daily Loss Limit Exceeded
    await run_scenario(
        scenario_name="2. Circuit Breaker - Daily Loss >= 2.5%",
        user_id="user_breached_loss",
        query="Perform the pre-market scan and generate risk-sized bracket recommendations for user_breached_loss"
    )

    # Scenario 3: Position Cap Breached
    await run_scenario(
        scenario_name="3. Position Cap - Already Holding 3+ Positions",
        user_id="user_max_positions",
        query="Perform the pre-market scan and generate risk-sized bracket recommendations for user_max_positions"
    )

    # Scenario 4: User Order Approval Flow
    await run_scenario(
        scenario_name="4. User Order Approval - Submit AAPL Bracket Order",
        user_id="user_happy",
        query="I approve the AAPL recommendation. Please submit the bracket order for me."
    )


if __name__ == "__main__":
    asyncio.run(main())
