# Review Checklist

## Structure

- Confirm real runtime entrypoint files.
- Confirm whether API service exists or only CLI exists.
- Confirm orchestrator graph topology and loop conditions.

## Agents and Tools

- Map each agent to allowed tools.
- Check whether prompts and node composition are aligned.
- Check tool result compacting and context-size controls.

## RAG and Persistence

- Verify ingest path, retrieval path, and vector backend.
- Verify DB pool/checkpointer/store initialization behavior.
- Verify fallback behavior when DSN or API keys are missing.

## Reliability

- Check presence of tests and minimal coverage targets.
- Check exception handling boundaries in CLI and workflow runtime.
- Check whether reflection loop can degrade into wasteful retries.

## Maintainability

- Identify repeated code blocks that should be shared.
- Identify stale docs that no longer match source code.
- Identify hardcoded knobs that should move to environment config.
