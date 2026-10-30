import json
import litellm
import subprocess
from rich import print
import re

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

def validate_messages(messages):
    prev = None
    for msg in messages:
        role = msg["role"]
        if role == "system":
            continue
        if role == "assistant" and prev not in (None, "user", "tool"):
            raise ValueError(f"'assistant' after '{prev}' — expected after 'user' or 'tool'")
        if role == "user" and prev not in (None, "assistant"):
            raise ValueError(f"'user' after '{prev}' — expected after 'assistant'")
        if role == "tool" and prev not in ("assistant", "tool"):
            raise ValueError(f"'tool' after '{prev}' — expected after 'assistant' or 'tool'")
        prev = role

def extract_bash(text):
    pattern = r'```bash(.+?)```'
    match = re.search(pattern, text, re.DOTALL)
    return match

system_prompt = """You are a coding assistant that solves tasks by running bash commands.

## How you work

1. Think about what to do
2. Use proper tool when needed
3. You'll see the result of the tool execution in the next message
4. Repeat until the task is done
5. Say DONE when finished

## Rules

- Each command runs in a fresh shell (cd and env vars don't persist)
- Keep commands simple — one step at a time
"""

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": "Find all python files in the current directory"}]

while True:
    validate_messages(messages)
    response = litellm.completion(model="anthropic/claude-haiku-5-5", messages=messages, tools=[BASH_TOOL])
    print("=== Assistant Response ===\n")
    print(response)
    message = response["choices"][0]["message"]
    reply = message.content or ""
    tool_calls = message.tool_calls
    print(f"=== Assistant reply ===\n{reply}\n")

    # Append the full assistant message (content + tool_calls)
    messages.append(message.model_dump())

    if not tool_calls:
        print("=== Task complete (no tool calls) ===")
        break

    for tool_call in tool_calls:
        args = json.loads(tool_call.function.arguments)
        bash_command = args.get("command")
        if not bash_command:
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": "Error: no command provided"})
            continue
        print(f"=== Running bash command ===\n{bash_command}\n")
        result = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
        output = f"returncode: {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        print(f"=== Bash command output ===\n{output}\n")
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
