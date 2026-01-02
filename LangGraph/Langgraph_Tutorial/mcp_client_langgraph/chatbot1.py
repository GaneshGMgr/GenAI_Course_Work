from model import llm_ollama

# Define calculator tool as normal Python function
def calculator(first_num: int, second_num: int, operation: str) -> dict:
    """Perform basic arithmetic operations"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            result = first_num / second_num if second_num != 0 else None
        elif operation == "mod":
            result = first_num % second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

# Available tools mapping
available_tools = {"calculator": calculator}

# Start conversation
messages = [("system", "You are a cricket commentator. Solve math problems and comment like cricket commentary.")]
messages.append(("human", "Find the modulus of 132354 and 23 and give answer like a cricket commentator."))

# Manual loop to handle tool calls
while True:
    response = llm_ollama.invoke(messages)  # Invoke ChatOllama
    messages.append(("assistant", response.content))
    
    print("Assistant says:", response.content)

    # Check for tool calls (LangChain-style, simulated here)
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            if name in available_tools:
                result = available_tools[name](**args)
                print(f"Tool {name} called with {args}, result: {result}")
                # Append tool result back to conversation
                messages.append(("tool", str(result)))
    else:
        # No tool calls remaining, stop the loop
        break

# Final answer
print("\n=== Final Commentator Response ===")
print(messages[-1][1])
