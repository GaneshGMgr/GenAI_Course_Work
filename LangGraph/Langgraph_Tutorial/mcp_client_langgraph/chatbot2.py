from ollama import chat, ChatResponse
from model import llm_chat_Openai
from model import MODEL_NAME

# Define the calculator tool as a normal Python function
def calculator(first_num: int, second_num: int, operation: str) -> dict:
    """Perform basic arithmetic operations."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        elif operation == "mod":
            result = first_num % second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

# Map of available tools
available_tools = {
    "calculator": calculator
}

# Start conversation
messages = [{"role": "user", "content": "Find the modulus of 132354 and 23 and give answer like a cricket commentator."}]

while True:
    # Ask the model
    response: ChatResponse = chat(
        model=MODEL_NAME,
        messages=messages,
        tools=[calculator],
        think=True  # enables multi-step reasoning
    )
    
    # Append model message
    messages.append(response.message)
    
    # Print thinking / content
    print("Thinking:", response.message.thinking)
    print("Content:", response.message.content)
    
    # Execute any tool calls from the model
    if response.message.tool_calls:
        for tc in response.message.tool_calls:
            tool_name = tc.function.name
            if tool_name in available_tools:
                args = tc.function.arguments
                print(f"Calling {tool_name} with arguments {args}")
                result = available_tools[tool_name](**args)
                print(f"Result: {result}")
                # Add the tool result back into the conversation
                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(result)
                })
    else:
        # No more tool calls, stop the loop
        break

# Final model response
print("\n=== Final Response ===")
print(messages[-1].content)
