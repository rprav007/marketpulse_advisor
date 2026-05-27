import json
import logging
import re
from a2a import types
from a2a import utils
from a2a.server import agent_execution
from a2a.server import events
from a2a.server import tasks
from a2a.utils import errors as a2a_errors
from google.adk import runners
from google.adk.artifacts import in_memory_artifact_service
from google.adk.memory import in_memory_memory_service
from google.adk.sessions import in_memory_session_service
from google.genai import types as genai_types
import jsonschema

import a2ui_schema
from .agent import root_agent

logger = logging.getLogger(__name__)


class AdkAgentToA2AExecutor(agent_execution.AgentExecutor):
  """An agent executor for ADK agents with A2UI parsing and validation."""

  _runner: runners.Runner

  def __init__(self):
    # Prepare A2UI schema validator
    try:
      single_message_schema = json.loads(a2ui_schema.A2UI_SCHEMA)
      self.a2ui_schema_object = {
          "type": "array",
          "items": single_message_schema,
      }
      logger.info("[DEBUG] A2UI_SCHEMA successfully loaded.")
    except Exception as e:
      logger.error("[DEBUG] Failed to parse A2UI_SCHEMA: %s", e)
      self.a2ui_schema_object = None

    self._agent = root_agent
    self._runner = runners.Runner(
        app_name=self._agent.name,
        agent=self._agent,
        session_service=in_memory_session_service.InMemorySessionService(),
        artifact_service=in_memory_artifact_service.InMemoryArtifactService(),
        memory_service=in_memory_memory_service.InMemoryMemoryService(),
    )
    self._user_id = "remote_agent"

  async def execute(
      self,
      context: agent_execution.RequestContext,
      event_queue: events.EventQueue,
  ) -> None:
    query = context.get_user_input()
    task = context.current_task
    logger.info("[DEBUG] Query: %s", query)

    if not task:
      if not context.message:
        return

      task = utils.new_task(context.message)
      await event_queue.enqueue_event(task)

    updater = tasks.TaskUpdater(event_queue, task.id, task.context_id)
    session_id = task.context_id

    session = await self._runner.session_service.get_session(
        app_name=self._agent.name,
        user_id=self._user_id,
        session_id=session_id,
    )
    if session is None:
      session = await self._runner.session_service.create_session(
          app_name=self._agent.name,
          user_id=self._user_id,
          state={},
          session_id=session_id,
      )

    # SESSION RECOVERY & CONTEXT OVERRIDE: Extract state from A2UI payload
    try:
      if hasattr(context, "message") and context.message:
        for part in context.message.parts:
          if hasattr(part, "root") and hasattr(part.root, "data"):
            data = part.root.data
            if isinstance(data, dict) and "userAction" in data:
              action_ctx = data["userAction"].get("context", {})
              query = action_ctx.get("message", query)
              # Recover inputs from forms
              for item in data["userAction"].get("inputs", []):
                if item.get("id"):
                  context.metadata[item["id"]] = item["value"]
    except Exception as e:
      logger.warning(f"A2UI session recovery failed: {e}")

    # STATE INJECTION (Transcript Echoing)
    state_str = "|".join([f"{k}={v}" for k, v in context.metadata.items()])
    if state_str:
      query = f"{query} [State: {state_str}]"

    current_query_text = query
    max_retries = 1
    attempt = 0

    # Working status
    await updater.start_work()

    while attempt <= max_retries:
      attempt += 1
      content = genai_types.Content(
          role="user", parts=[{"text": current_query_text}]
      )

      final_response_content = None
      logger.info("[DEBUG] attempt: %s", attempt)

      try:
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
          if event.is_final_response():
            if (
                event.content
                and event.content.parts
                and event.content.parts[0].text
            ):
              final_response_content = "\n".join(
                  [p.text for p in event.content.parts if p.text]
              )
              logger.info(
                  "[DEBUG] Final response content: %s", final_response_content
              )

      except Exception as e:
        await updater.failed(
            message=utils.new_agent_text_message(
                f"Task failed with error: {str(e)}"
            )
        )
        return

      if final_response_content is None:
        if attempt <= max_retries:
          current_query_text = "I received no response. Please try again."
          continue
        else:
          await updater.failed(
              message=utils.new_agent_text_message("No response generated.")
          )
          return

      logger.info("[DEBUG] Final response content: %s", final_response_content)
      is_valid = False
      error_message = ""
      json_string_cleaned = "[]"
      text_part = final_response_content

      if "---a2ui_JSON---" not in final_response_content:
        error_message = "Delimiter '---a2ui_JSON---' not found."
      else:
        try:
          text_part, json_string = final_response_content.split(
              "---a2ui_JSON---", 1
          )
          # Clean json_string (stripping ```json blocks if LLM adds them)
          json_string_cleaned = (
              json_string.strip().lstrip("```json").rstrip("```").strip()
          )

          if not json_string_cleaned:
            json_string_cleaned = "[]"

          parsed_json = json.loads(json_string_cleaned)
          logger.info("[DEBUG] Parsed JSON: %s", parsed_json)

          # Ensure the LLM wraps the list of messages under a top-level `"a2ui_messages"` key
          if isinstance(parsed_json, dict) and "a2ui_messages" in parsed_json:
            parsed_json_list = parsed_json["a2ui_messages"]
          else:
            # Fallback if LLM output a raw list
            parsed_json_list = parsed_json if isinstance(parsed_json, list) else [parsed_json]

          if self.a2ui_schema_object:
            jsonschema.validate(
                instance=parsed_json_list, schema=self.a2ui_schema_object
            )

          is_valid = True
        except Exception as e:
          error_message = f"Validation failed: {str(e)}"

      if is_valid:
        parts = []
        if text_part.strip():
          parts.append(types.Part(root=types.TextPart(text=text_part.strip())))

        logger.info("[DEBUG] UI JSON: %s", json_string_cleaned)

        parsed_json = json.loads(json_string_cleaned)
        if isinstance(parsed_json, dict) and "a2ui_messages" in parsed_json:
          json_data_list = parsed_json["a2ui_messages"]
        else:
          json_data_list = parsed_json if isinstance(parsed_json, list) else [parsed_json]

        # Send each message as a separate DataPart
        for message in json_data_list:
          ui_data_part = types.Part(
              root=types.DataPart(
                  data=message,
                  metadata={"mimeType": "application/json+a2ui"},
              )
          )
          parts.append(ui_data_part)

        logger.info("[DEBUG] Parts: %s", parts)

        await updater.add_artifact(parts, name="response")
        await updater.complete()
        return

      else:
        if attempt <= max_retries:
          current_query_text = (
              f"Your previous response was invalid: {error_message}. You MUST"
              " generate a valid response that strictly follows the A2UI JSON"
              " SCHEMA. Output the conversational response first, then the delimiter "
              "'---a2ui_JSON---', followed by the JSON object wrapped with a top-level 'a2ui_messages' key."
              f" Please retry the original request: '{query}'"
          )
          logger.warning(
              "[DEBUG] Retrying due to validation error: %s", error_message
          )
          continue
        else:
          await updater.add_artifact(
              [
                  types.Part(
                      root=types.TextPart(
                          text=(
                              "I encountered an error generating the UI:"
                              f" {error_message}. Here is the raw response:"
                              f" {final_response_content}"
                          )
                      )
                  )
              ],
              name="error_response",
          )
          await updater.complete()
          return

  async def cancel(
      self,
      context: agent_execution.RequestContext,
      event_queue: events.EventQueue,
  ) -> None:
    raise a2a_errors.ServerError(error=types.UnsupportedOperationError())
