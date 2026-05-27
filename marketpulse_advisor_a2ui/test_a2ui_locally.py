import os
import sys
import json
import asyncio
import jsonschema

# Add parent dir to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import Runner
from google.genai import types as genai_types
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from marketpulse_advisor_a2ui.agent import root_agent
import marketpulse_advisor_a2ui.a2ui_schema as a2ui_schema
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)


async def test_scenario(user_id, query_text):
  print("\n" + "=" * 80)
  print(f"RUNNING SCENARIO FOR {user_id}...")
  print(f"Query: {query_text}")
  print("=" * 80)

  # Instantiate runner
  runner = Runner(
      app_name="MarketPulseAdvisorA2UI_LocalTest",
      agent=root_agent,
      session_service=InMemorySessionService(),
      artifact_service=InMemoryArtifactService(),
      memory_service=InMemoryMemoryService(),
  )

  # Create session
  session = await runner.session_service.create_session(
      app_name="MarketPulseAdvisorA2UI_LocalTest",
      user_id="local_test_user",
      state={},
  )


  # Run agent
  message = genai_types.Content(parts=[{"text": query_text}])
  full_response = ""

  async for event in runner.run_async(
      user_id="local_test_user", session_id=session.id, new_message=message
  ):
    if event.is_final_response():
      for part in event.content.parts:
        if hasattr(part, "text") and part.text:
          full_response += part.text

  print("\n--- AGENT RESPONSE ---")
  print(full_response)
  print("----------------------")

  # Validate A2UI output structure
  if "---a2ui_JSON---" not in full_response:
    print("❌ ERROR: A2UI delimiter '---a2ui_JSON---' not found in response.")
    return False

  text_part, json_part = full_response.split("---a2ui_JSON---", 1)
  json_part_cleaned = (
      json_part.strip().lstrip("```json").rstrip("```").strip()
  )

  print("\nParsed Text Part:")
  print(text_part.strip())

  print("\nParsed A2UI JSON Part:")
  print(json_part_cleaned)

  try:
    parsed_json = json.loads(json_part_cleaned)
    print("✓ Valid JSON parsing successful.")

    # Ensure the top-level messages key exists
    if not isinstance(parsed_json, dict) or "a2ui_messages" not in parsed_json:
      print(
          "❌ ERROR: JSON is missing the top-level 'a2ui_messages' object"
          " wrapper."
      )
      return False
    print("✓ Top-level 'a2ui_messages' key is present.")

    # Load and validate against schema
    single_schema = json.loads(a2ui_schema.A2UI_SCHEMA)
    list_schema = {"type": "array", "items": single_schema}

    jsonschema.validate(instance=parsed_json["a2ui_messages"], schema=list_schema)
    print("✓ A2UI JSON validated successfully against v0.8 schema.")
    print("🎉 SUCCESS!")
    return True
  except json.JSONDecodeError as je:
    print(f"❌ ERROR: Failed to parse A2UI JSON: {je}")
    return False
  except jsonschema.exceptions.ValidationError as ve:
    print(f"❌ ERROR: A2UI schema validation failed: {ve}")
    return False


async def main():
  # Setup dummy environment vars if not already set
  os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "rad-alm-test")
  os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

  # Run Scenario 1 (Happy Path)
  success1 = await test_scenario(
      "user_happy", "Perform the pre-market scan for user_happy"
  )

  # Run Scenario 3 (Max Positions Warning)
  success2 = await test_scenario(
      "user_max_positions",
      "Check for breakout recommendations for user_max_positions",
  )

  if success1 and success2:
    print("\n✅ All local A2UI test scenarios passed successfully!")
    sys.exit(0)
  else:
    print("\n❌ One or more A2UI test scenarios failed.")
    sys.exit(1)


if __name__ == "__main__":
  asyncio.run(main())
