import asyncio
import os
import sys
from typing import Any
from .utils.support_tool import support_ticket_agent



async def run_support_ticket(content: str) -> dict[str, Any]:
    """Run the support ticket workflow and return the final state."""
    state = {"content": content}
    return await support_ticket_agent.ainvoke(state)


if __name__ == "__main__":
    print("**** Please enter your question ****", flush=True)
    words = input()
    if not words:
        raise SystemExit("Ticket content is required.")
    print("**** Processing your request ****", flush=True)
    result = asyncio.run(support_ticket_agent.ainvoke({"content": words}))
    print(result.get("response", "No response generated."))
