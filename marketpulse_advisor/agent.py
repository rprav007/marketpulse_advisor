"""Agent definitions and orchestration for the MarketPulse Advisor system.

This module defines the specialized agents (Scanner, Backtester, Risk Advisor)
and the root Supervisor orchestrating them using the Google ADK.
"""

from __future__ import annotations

import logging
import os
import subprocess
from functools import cached_property

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.genai import Client, types
from google.oauth2.credentials import Credentials

from .tools import (
    get_market_scans,
    run_backtest,
    get_portfolio_data,
    submit_bracket_order
)

# Setup logging
logger = logging.getLogger("marketpulse_advisor.agent")


# --- Custom LLM Wrapper to inject gcloud access token credentials ---
class AuthedGemini(Gemini):
    """Gemini wrapper supporting API keys, Application Default Credentials, and gcloud token fallbacks."""
    
    @cached_property
    def api_client(self) -> Client:
        try:
            from .tools import reload_env
            reload_env()
        except Exception:
            pass

        # 1. Prioritize Developer API Key if present
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            logger.info("Initializing GenAI client using GEMINI_API_KEY.")
            return Client(api_key=api_key)

        # 2. Check if Application Default Credentials (ADC) are configured
        adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        default_adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        if adc_path or os.path.exists(default_adc):
            logger.info("Initializing GenAI client using Application Default Credentials (ADC).")
            return Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT", "rad-alm-test"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            )

        # 3. Fallback to gcloud access token print
        try:
            logger.info("ADC/API key missing. Attempting user credentials token fetch via gcloud...")
            logger.warning(
                "*** WARNING: Falling back to 'gcloud auth print-access-token'. ***\n"
                "Note that Vertex AI typically rejects personal user access tokens (resulting in "
                "401 UNAUTHENTICATED / ACCESS_TOKEN_TYPE_UNSUPPORTED).\n"
                "To fix this, please run 'gcloud auth application-default login' in your terminal "
                "to set up Application Default Credentials, or add your Google AI Studio developer "
                "key 'GEMINI_API_KEY' to your .env file."
            )
            token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"], 
                text=True
            ).strip()
            creds = Credentials(token)
            return Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT", "rad-alm-test"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                credentials=creds
            )
        except Exception as e:
            logger.warning(f"Could not retrieve gcloud credentials: {e}. Falling back to default ADK client.")
            return super().api_client

    @cached_property
    def _live_api_client(self) -> Client:
        try:
            from .tools import reload_env
            reload_env()
        except Exception:
            pass

        # 1. Prioritize Developer API Key
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            logger.info("Initializing GenAI live client using GEMINI_API_KEY.")
            return Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    api_version=self._live_api_version,
                )
            )

        # 2. Check if Application Default Credentials (ADC) are configured
        adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        default_adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        if adc_path or os.path.exists(default_adc):
            logger.info("Initializing GenAI live client using Application Default Credentials (ADC).")
            return Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT", "rad-alm-test"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                http_options=types.HttpOptions(
                    api_version=self._live_api_version,
                )
            )

        # 3. Fallback to gcloud access token print
        try:
            logger.info("ADC/API key missing. Attempting user credentials token fetch via gcloud...")
            logger.warning(
                "*** WARNING: Falling back to 'gcloud auth print-access-token'. ***\n"
                "Note that Vertex AI typically rejects personal user access tokens (resulting in "
                "401 UNAUTHENTICATED / ACCESS_TOKEN_TYPE_UNSUPPORTED).\n"
                "To fix this, please run 'gcloud auth application-default login' in your terminal "
                "to set up Application Default Credentials, or add your Google AI Studio developer "
                "key 'GEMINI_API_KEY' to your .env file."
            )
            token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"], 
                text=True
            ).strip()
            creds = Credentials(token)
            return Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT", "rad-alm-test"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                credentials=creds,
                http_options=types.HttpOptions(
                    api_version=self._live_api_version,
                )
            )
        except Exception as e:
            logger.warning(f"Could not retrieve live gcloud credentials: {e}. Falling back to default ADK client.")
            return super()._live_api_client



# Instantiate our authed model
MODEL = AuthedGemini(model="gemini-2.5-flash")

# --- 1. MarketScannerAgent ---
scanner_agent = Agent(
    name="MarketScannerAgent",
    model=MODEL,
    description="Scans the broader market for stocks undergoing Bollinger Band squeezes and high-volume breakout patterns.",
    instruction=(
        "You are a technical market scanner. Your only job is to scan the market "
        "for volatility compression breakouts by running the 'get_market_scans' tool. "
        "Do not guess or generate candidates yourself. Run the tool and return the exact "
        "list of symbols, prices, ATRs, and sectors. Explain that these candidates are "
        "primed for 2-to-3-day breakout expansions."
    ),
    tools=[get_market_scans]
)

# --- 2. BacktestAgent ---
backtest_agent = Agent(
    name="BacktestAgent",
    model=MODEL,
    description="Validates breakout setups by executing historical backtests over a 10-year period.",
    instruction=(
        "You are a backtesting validator. Your job is to check the 72-hour forward "
        "backtest metrics (win-rate, average drawdown, and average upside) for each stock symbol. "
        "Read the list of symbols found by MarketScannerAgent from the previous turn/context. "
        "For each symbol, execute the 'run_backtest' tool. Present the compiled statistics "
        "clearly to show the historical probability of success."
    ),
    tools=[run_backtest]
)

# --- 3. RiskAdvisorAgent ---
risk_advisor_agent = Agent(
    name="RiskAdvisorAgent",
    model=MODEL,
    description="Evaluates user portfolio states, enforces circuit breakers, and calculates risk-bracket parameters.",
    instruction=(
        "You are a strict risk management expert. Your job is to perform position sizing and safety checks "
        "for a specific subscriber (using their user_id). "
        "Use the candidates and backtest results identified in the previous steps to execute the safety checks and position sizing.\n\n"
        "You must execute the following sequential rules:\n\n"
        "1. Fetch the user's current account balance and status by calling the 'get_portfolio_data' tool.\n"
        "2. Check Daily Loss Limit: If the daily PnL shows a loss greater than or equal to 2.5% of "
        "total account equity (e.g. daily PnL <= -0.025 * equity), raise a critical alert and REFUSE to recommend trades.\n"
        "3. Check Open Positions Cap: If the user already has 3 or more open positions, proceed with calculating and recommending the trade setups, but explicitly include a WARNING indicating that the subscriber has reached the recommended 3-position cap.\n"
        "4. Check Sector Exposure Cap: For each candidate setup, check if the candidate's sector already exists "
        "in the user's active sector weights. If it does, ensure adding this new position (capped at 20% of account equity) "
        "does not exceed a total of 20% exposure for that sector. Discard candidates that breach the 20% sector limit.\n"
        "5. Risk-of-Ruin Sizing: For candidates passing safety checks (including those that proceed with a 3-position cap warning), compute the custom share size (S) using the formula:\n"
        "   S = (E * R) / (Pe - Ps)\n"
        "   - E: Total Account Equity\n"
        "   - R: Risk percentage (use exactly 1.0% of total equity, i.e., 0.01 * equity)\n"
        "   - Pe: Entry Price (breakout price from the candidate data)\n"
        "   - Ps: Stop-Loss Price (calculated as Pe - (1.5 * ATR))\n"
        "   - Take-Profit Target (Pt): calculated as Pe + (4.5 * ATR) to enforce a minimum 1:3 Risk-to-Reward ratio.\n"
        "6. Return the calculated share count (S) rounded down to the nearest integer, along with the precise entry, "
        "stop-loss, and take-profit targets for each approved candidate stock. Present your calculations explicitly and state if any warning applies."
    ),
    tools=[get_portfolio_data]
)

# --- Premarket Workflow Agent (Sequential Workflow) ---
premarket_workflow = SequentialAgent(
    name="PreMarketWorkflow",
    sub_agents=[scanner_agent, backtest_agent, risk_advisor_agent]
)

# --- 4. MarketPulseSupervisor (Root Orchestrator) ---
root_agent = Agent(
    name="MarketPulseSupervisor",
    model=MODEL,
    description="Orchestrator for the MarketPulse Advisor multi-agent SaaS pipeline. Coordinates scanners, backtesters, and risk logic.",
    instruction=(
        "You are the MarketPulse Supervisor Agent, the main orchestrator of the MarketPulse Advisor system. "
        "Your task is to coordinate the technical scanning, historical validation, and custom risk sizing flow "
        "to generate tailored pre-market trading recommendations for a subscriber.\n\n"
        "Upon initial conversation, you should orient the user: introduce yourself, state your capabilities, "
        "and ask for the trader's SaaS subscriber ID (user_id) if it is not already in the session state.\n\n"
        "When the pipeline is run (either pre-market or when requested with a user_id):\n"
        "1. First, verify you have the trader's user_id. If not, ask the user for it first.\n"
        "2. Transfer control to the `PreMarketWorkflow` sub-agent to execute the pre-market scan, backtesting, and risk checks.\n"
        "3. Once the `PreMarketWorkflow` completes, summarize the final recommendations to the user, showing "
        "their symbol, backtest win-rate, custom share size, entry limit, stop-loss, and take-profit targets.\n"
        "4. If the trader explicitly approves a recommended trade setup, execute the trade by calling the "
        "'submit_bracket_order' tool with the user_id, symbol, calculated quantity, and price targets. "
        "Confirm the OCO order submission to the user."
    ),
    tools=[submit_bracket_order],
    sub_agents=[premarket_workflow]
)
