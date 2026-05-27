import os
import google.cloud.aiplatform as aiplatform

# Pre-initialize project ID to bypass the project number DNS lookup failure in Agent Engine replicas
if "PROJECT_ID" in os.environ:
    aiplatform.init(project=os.environ["PROJECT_ID"])
    os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ["PROJECT_ID"]

from vertexai.preview.reasoning_engines import A2aAgent
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card
from a2a.types import AgentSkill
from a2ui.a2a import get_a2ui_agent_extension
from .agent_executor import AdkAgentToA2AExecutor

skill = AgentSkill(
    id="marketpulse_advisor",
    name="MarketPulse Advisor Agent",
    description="Multi-agent pre-market scanner, backtester, and risk advisor.",
    tags=["finance", "scanner", "risk-advisor"],
    examples=["Perform the pre-market scan for user_happy"]
)

agent_card = create_agent_card(
    agent_name="MarketPulse Advisor",
    description="Multi-agent pre-market scanner, backtester, and risk advisor with A2UI dashboard rendering.",
    skills=[skill],
    streaming=True
)

a2ui_extension = get_a2ui_agent_extension(
    version="0.8",
    accepts_inline_catalogs=False,
    supported_catalog_ids=["https://a2ui.org/specification/v0_8/standard_catalog_definition.json"]
)

agent_card.capabilities.extensions = [a2ui_extension]

agent = A2aAgent(
    agent_card=agent_card,
    agent_executor_builder=AdkAgentToA2AExecutor
)
