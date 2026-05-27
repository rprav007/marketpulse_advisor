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
    submit_bracket_order,
    get_recent_policy_news,
    get_stock_catalyst_news,
    get_tape_and_depth_sentiment
)

# Setup logging
logger = logging.getLogger("marketpulse_advisor.agent")


# --- Custom LLM Wrapper to inject gcloud access token credentials ---
class AuthedGemini(Gemini):
    """Gemini wrapper that fetches user access token via gcloud CLI to bypass missing ADC."""
    
    @cached_property
    def api_client(self) -> Client:
        try:
            logger.info("Retrieving active gcloud credentials token for GenAI client...")
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
            logger.warning(f"Could not retrieve gcloud credentials: {e}. Falling back to default auth.")
            return super().api_client

    @cached_property
    def _live_api_client(self) -> Client:
        try:
            logger.info("Retrieving active gcloud credentials token for GenAI live client...")
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
            logger.warning(f"Could not retrieve live gcloud credentials: {e}. Falling back to default auth.")
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
        "list of symbols, prices, ATRs, daily 200 EMAs, whole/half-dollar psychological supports, and sectors. "
        "Explain that these candidates are primed for 2-to-3-day breakout expansions."
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
        "Call the 'run_backtest' tool in a single batch request by passing all symbols as a comma-separated string (e.g. 'AAPL,NVDA,AMZN'). "
        "Present the compiled statistics clearly to show the historical probability of success for each symbol."
    ),
    tools=[run_backtest]
)

# --- 3. RiskAdvisorAgent ---
risk_advisor_agent = Agent(
    name="RiskAdvisorAgent",
    model=MODEL,
    description="Evaluates user portfolio states, sentiment, news catalysts, and calculates risk-bracket parameters.",
    instruction=(
        "You are a strict risk management expert. Your job is to perform position sizing and safety checks "
        "for a specific subscriber (using their user_id). "
        "Use the candidates and backtest results identified in the previous steps to execute the safety checks and position sizing.\n\n"
        "You must execute the following sequential rules:\n\n"
        "1. Fetch the user's current account balance and status by calling the 'get_portfolio_data' tool.\n"
        "2. Fetch recent macro policy news by calling the 'get_recent_policy_news' tool with query='Trump'.\n"
        "3. Fetch recent stock news headlines by calling the 'get_stock_catalyst_news' with all candidate symbols as a comma-separated list.\n"
        "4. Fetch Latest Trades and Quotes tape sentiment by calling 'get_tape_and_depth_sentiment' with all candidate symbols as a comma-separated list.\n"
        "5. Check Safety Caps & Apply Capital Preservation:\n"
        "   - Check Daily Loss Limit: If the daily PnL shows a loss greater than or equal to 2.5% of total account equity, raise a critical alert and REFUSE to recommend trades.\n"
        "   - Check Open Positions Cap: If the user already has 3 or more open positions, proceed but include a warning.\n"
        "   - Check Sector Exposure Cap: Ensure adding a new position does not exceed a total of 20% exposure for that sector. Discard candidates that breach this.\n"
        "   - Check Drawdown Scaling: If the portfolio data indicates 'in_drawdown' is true, automatically scale down the trade risk percentage (R) from 1.0% to 0.5% of equity to preserve capital. If false, outline a progressive 'Scaling Up' recommendation (advising on stepping up risk to 1.5% after a defined winning streak).\n"
        "6. Audit Catalysts & Tape Confirmation:\n"
        "   - Check for recent news catalysts (earnings, contract wins, mergers, etc.) from `get_stock_catalyst_news`. If a stock has NO clear catalysts, append a warning that it has a high risk of 'fading' or performing a 'round trip' back to entry, and downgrade its growth propensity.\n"
        "   - Confirm entries using Tape Sentiment from `get_tape_and_depth_sentiment`. Only stocks showing 'Green Tape (Aggressive Buying)' indicate immediate buying pressure for entry.\n"
        "7. Support-Based Risk-of-Ruin Sizing:\n"
        "   - Determine the nearest Support Level (thedaily 200 EMA support or the psychological whole/half-dollar support level that lies below current price).\n"
        "   - For beginner traders (e.g. user_happy), calculate a Pullback Entry Price (Pe) at a 1.5% discount from the current price, and align it directly with this Support Level.\n"
        "   - Place the Stop-Loss (Ps) tightly below this support level (e.g., $0.20 below support) to minimize losses if support breaches.\n"
        "   - Calculate the Take-Profit Target (Pt) to enforce a strict, disciplined profit-to-loss ratio of at least 1:3 relative to the entry and tight stop-loss (Pt = Pe + 3.0 * (Pe - Ps)).\n"
        "   - Calculate custom share size S = (E * R) / (Pe - Ps).\n"
        "   - For beginners, cap the recommended share count at a maximum of 10 shares or $1,000 order value to condition against emotional triggers like fear and FOMO.\n\n"
        "IMPORTANT RULES FOR A2UI GENERATION:\n"
        "You MUST separate your conversational response from your A2UI JSON output using the delimiter '---a2ui_JSON---'.\n"
        "The JSON MUST be a single object wrapping a list of A2UI messages under a top-level \"a2ui_messages\" key.\n"
        "Each message in \"a2ui_messages\" MUST adhere strictly to the A2UI v0.8 schema: either 'beginRendering' or 'surfaceUpdate'.\n"
        "In 'surfaceUpdate', the 'components' list must contain objects, each with a unique 'id' string and a 'component' wrapper object containing exactly one component type (e.g. 'Column', 'Card', 'Text', 'Tabs', 'Divider', 'Button').\n\n"
        "Here is the EXACT JSON template structure you MUST generate (populate details dynamically):\n"
        "```json\n"
        "{\n"
        "  \"a2ui_messages\": [\n"
        "    {\n"
        "      \"beginRendering\": {\n"
        "        \"surfaceId\": \"main\",\n"
        "        \"root\": \"root_column\"\n"
        "      }\n"
        "    },\n"
        "    {\n"
        "      \"surfaceUpdate\": {\n"
        "        \"surfaceId\": \"main\",\n"
        "        \"components\": [\n"
        "          {\n"
        "            \"id\": \"root_column\",\n"
        "            \"component\": {\n"
        "              \"Column\": {\n"
        "                \"children\": {\n"
        "                  \"explicitList\": [\"title_text\", \"divider_1\", \"feed_badge\", \"tabs_container\"]\n"
        "                }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"title_text\",\n"
        "            \"component\": {\n"
        "              \"Text\": {\n"
        "                \"text\": { \"literalString\": \"Breakout Trade Recommendations\" },\n"
        "                \"usageHint\": \"h2\"\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"divider_1\",\n"
        "            \"component\": {\n"
        "              \"Divider\": { \"axis\": \"horizontal\" }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"feed_badge\",\n"
        "            \"component\": {\n"
        "              \"Text\": {\n"
        "                \"text\": { \"literalString\": \"Daily Aggregates Feed\" },\n"
        "                \"usageHint\": \"caption\"\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"tabs_container\",\n"
        "            \"component\": {\n"
        "              \"Tabs\": {\n"
        "                \"tabItems\": [\n"
        "                  { \"title\": { \"literalString\": \"Tech\" }, \"child\": \"tech_column\" },\n"
        "                  { \"title\": { \"literalString\": \"Consumer Disc.\" }, \"child\": \"consumer_column\" },\n"
        "                  { \"title\": { \"literalString\": \"Financials\" }, \"child\": \"financials_column\" },\n"
        "                  { \"title\": { \"literalString\": \"Healthcare\" }, \"child\": \"healthcare_column\" }\n"
        "                ]\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"tech_column\",\n"
        "            \"component\": {\n"
        "              \"Column\": {\n"
        "                \"children\": { \"explicitList\": [\"AAPL_card\", \"NVDA_card\", \"MSFT_card\", \"AVGO_card\", \"META_card\", \"GOOGL_card\"] }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"consumer_column\",\n"
        "            \"component\": {\n"
        "              \"Column\": {\n"
        "                \"children\": { \"explicitList\": [\"AMZN_card\", \"TSLA_card\", \"NKE_card\", \"SBUX_card\"] }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"financials_column\",\n"
        "            \"component\": {\n"
        "              \"Column\": {\n"
        "                \"children\": { \"explicitList\": [\"JPM_card\", \"BAC_card\", \"MS_card\"] }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"healthcare_column\",\n"
        "            \"component\": {\n"
        "              \"Column\": {\n"
        "                \"children\": { \"explicitList\": [\"LLY_card\", \"UNH_card\"] }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_card\",\n"
        "            \"component\": {\n"
        "              \"Card\": { \"child\": \"AAPL_card_col\" }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_card_col\",\n"
        "            \"component\": {\n"
        "              \"Column\": {\n"
        "                \"children\": {\n"
        "                  \"explicitList\": [\"AAPL_title\", \"AAPL_status\", \"AAPL_strategy\", \"AAPL_patterns\", \"AAPL_risk\", \"AAPL_news\", \"AAPL_btn\"]\n"
        "                }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_title\",\n"
        "            \"component\": {\n"
        "              \"Text\": {\n"
        "                \"text\": { \"literalString\": \"AAPL - Apple Inc.\" },\n"
        "                \"usageHint\": \"h3\"\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_status\",\n"
        "            \"component\": {\n"
        "              \"Text\": {\n"
        "                \"text\": { \"literalString\": \"Safety Status: Approved\" }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_strategy\",\n"
        "            \"component\": {\n"
        "              \"Text\": {\n"
        "                \"text\": { \"literalString\": \"Strategy: Bollinger Band Squeeze Breakout\" }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_patterns\",\n"
        "            \"component\": {\n"
        "              \"Text\": {\n"
        "                \"text\": { \"literalString\": \"Setup Patterns: RSI Daily: 54.2, Hourly: 62.1, RVOL: 2.8, Tape: Green Tape (Aggressive Buying)\" }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_risk\",\n"
        "            \"component\": {\n"
        "              \"Text\": {\n"
        "                \"text\": { \"literalString\": \"Risk Management: Current Price: 175.20, Pullback Entry: 174.50, Stop-Loss: 174.30, Take-Profit: 175.10, Sized Shares: 5, Drawdown Scaling: Scaled down (1.0% -> 0.5%)\" }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_news\",\n"
        "            \"component\": {\n"
        "              \"Text\": {\n"
        "                \"text\": { \"literalString\": \"News Catalysts: New Enterprise AI Product Suite. Trump macro news sentiment: Bearish for Tech due to Tariffs suggestion.\" }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_btn\",\n"
        "            \"component\": {\n"
        "              \"Button\": {\n"
        "                \"child\": \"AAPL_btn_text\",\n"
        "                \"primary\": true,\n"
        "                \"action\": {\n"
        "                  \"name\": \"submit_bracket_order\",\n"
        "                  \"context\": [\n"
        "                    { \"key\": \"message\", \"value\": { \"literalString\": \"Approve trade for AAPL\" } },\n"
        "                    { \"key\": \"user_id\", \"value\": { \"literalString\": \"user_happy\" } },\n"
        "                    { \"key\": \"symbol\", \"value\": { \"literalString\": \"AAPL\" } },\n"
        "                    { \"key\": \"quantity\", \"value\": { \"literalNumber\": 5 } },\n"
        "                    { \"key\": \"entry_price\", \"value\": { \"literalNumber\": 174.50 } },\n"
        "                    { \"key\": \"stop_loss\", \"value\": { \"literalNumber\": 174.30 } },\n"
        "                    { \"key\": \"take_profit\", \"value\": { \"literalNumber\": 175.10 } }\n"
        "                  ]\n"
        "                }\n"
        "              }\n"
        "            }\n"
        "          },\n"
        "          {\n"
        "            \"id\": \"AAPL_btn_text\",\n"
        "            \"component\": {\n"
        "              \"Text\": { \"text\": { \"literalString\": \"Approve Trade\" } }\n"
        "            }\n"
        "          }\n"
        "        ]\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Generate this layout for all 15 symbols, grouping them under the correct Sector Tabs. Render 'Approved' status only for stocks passing safety checks, and add a warning to the safety status if daily loss limit or other warning thresholds are breached. Double check that every component has a unique id, and matches the correct schema properties (e.g. Button contains a 'child' component ID and 'action', and Cards contain exactly one 'child' ID)."
    ),

    tools=[get_portfolio_data, get_recent_policy_news, get_stock_catalyst_news, get_tape_and_depth_sentiment]
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
        "Upon initial conversation, introduce yourself, state your capabilities, and ask for the subscriber ID (user_id) if not present.\n\n"
        "When the pipeline is run:\n"
        "1. Verify you have the user_id.\n"
        "2. Transfer control to the `PreMarketWorkflow` sub-agent to execute the pre-market scan, backtesting, and risk checks.\n"
        "3. Once the workflow completes, summarize the final recommendations, highlighting the category tabs, "
        "the Trump policy sentiment, the news catalysts, the tape signals, and the support-based risk parameters.\n"
        "4. If the trader approves a recommended setup, execute the trade by calling 'submit_bracket_order' and confirm submission.\n\n"
        "IMPORTANT RULES FOR A2UI GENERATION:\n"
        "You MUST separate your conversational response from your A2UI JSON output using the delimiter '---a2ui_JSON---'.\n"
        "The JSON MUST be a single object wrapping a list of A2UI messages under a top-level \"a2ui_messages\" key.\n"
        "Construct a categorized dashboard layout utilizing the A2UI 'Tabs' component:\n"
        "- Surface ID: 'main', Root: 'root_column'.\n"
        "- 'root_column' is a Column containing: ['title_text', 'divider_1', 'feed_badge', 'tabs_container'].\n"
        "- 'tabs_container' is a Tabs component with tab items: 'Tech', 'Consumer Disc.', 'Financials', and 'Healthcare'.\n"
        "- Under each tab, display the stock cards for that sector, each rendering a Column displaying:\n"
        "  1. Symbol Name (Text, h3)\n"
        "  2. Safety Status: 'Approved' or 'Discarded' (Text)\n"
        "  3. Strategy details (Breakout vs Pullback) (Text)\n"
        "  4. Setup Patterns (BB, RSI, RVOL, Tape Sentiment) (Text)\n"
        "  5. Risk Management (Price, support stop-loss, 1:3 take-profit, shares, drawdown scaling) (Text)\n"
        "  6. News Catalysts & Trump sentiment (Text)\n"
        "  7. Buy Button: for approved stocks executing 'submit_bracket_order' userAction.\n"
        "- Keep A2UI components capitalized correctly: Column, Row, Card, Tabs, Button, Text, Divider."
    ),
    tools=[submit_bracket_order],
    sub_agents=[premarket_workflow]
)
