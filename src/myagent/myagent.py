import litellm
import subprocess
from rich import print
import re

def validate_messages(messages):
    expected = "user"
    for msg in messages:
        if msg["role"] == "system":
            continue
        if msg["role"] != expected:
            raise ValueError(f"Expected '{expected}' but got '{msg['role']}'")
        expected = "user" if expected == "assistant" else "assistant"

def extract_bash(text):
    pattern = r'```bash(.+?)```'
    match = re.search(pattern, text, re.DOTALL)
    return match

system_prompt = """You are a coding assistant that solves tasks by running bash commands.                                                                       

## How you work                                                                                                                                                 

1. Think about what to do                                                                                                                                       
2. Respond with a bash command in a ```bash``` block                                                                                                       
3. You'll see the output                                                                                                                                        
4. Repeat until the task is done                                                                                                                                
5. Say DONE when finished                                                                                                                                       

## Rules                                                                                                                                                        

- Each response must contain at most one ```bash``` block                                                                                                                                                                                                                                                                                             
- Keep commands simple — one step at a time                                                                                                                     
"""

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": "Find all python files in the current directory"}]

while True:
    validate_messages(messages)
    response = litellm.completion(model="anthropic/claude-haiku-5-5", messages=messages)
    print("=== Assistant Response ===\n")
    print(response)
    reply = response["choices"][0]["message"]["content"] or ""
    print(f"=== Assistant reply ===\n{reply}\n")
    messages.append({"role": "assistant", "content": reply})

    if "DONE" in reply:
        print("=== Task complete ===")
        break

    match = extract_bash(reply)
    if match:
        bash_command = match.group(1).strip()
        print(f"=== Running bash command ===\n{bash_command}\n")
        result = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
        print(f"=== Bash command output ===\n{result.stdout}\n")
        output = f"returncode: {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        messages.append({"role": "user", "content": output})
    else:
        print("No bash command found in the assistant's reply.")
        break
