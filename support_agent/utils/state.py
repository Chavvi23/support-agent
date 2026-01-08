from typing import TypedDict


class TicketState(TypedDict):
    content: str
    category: str
    urgency: str
    sentiment: str
    summary: str
    doc_snippet: str
    action: str
    reason: str
    response: str
