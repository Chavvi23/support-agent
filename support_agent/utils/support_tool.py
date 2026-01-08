"""
Support Tool: LangGraph flow for processing support tickets.
"""

import asyncio
import json
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from support_agent.utils.helper import (
    chunk_text,
    compute_idf,
    cosine_similarity,
    extract_snippet,
    fetch_url_text,
    read_local_text,
    tfidf_vector,
    tokenize,
)
from support_agent.utils.models import llm_support
from support_agent.utils.state import TicketState
import logging

logger = logging.getLogger(__name__)



async def classify_ticket_step(state: TicketState, config: RunnableConfig) -> dict:
    """Step 1: Classify the ticket."""
    try:
        logger.info("Classifying support ticket")
        message = [
            {
                "role": "user",
                "content": (
                    "You are a support triage agent. Classify the ticket using these fields:\n"
                    "- category: one of billing, bug, account, feature request or other\n"
                    "- urgency: low, medium, or high based on customer impact and time-sensitivity\n"
                    "- sentiment: positive, neutral, or negative based on tone\n\n"
                    "IMPORTANT: You MUST return ONLY valid JSON in this exact format:\n"
                    '{"category": "billing", "urgency": "medium", "sentiment": "neutral"}\n'
                    "Do not include any other text, explanations, or markdown formatting.\n\n"
                    f"Ticket:\n{state['content']}"
                ),
            }
        ]
        response = await llm_support.ainvoke(message)
        data = json.loads(response.content)
        return {
            "category": data["category"],
            "urgency": data["urgency"],
            "sentiment": data["sentiment"],
        }
    except Exception as exc:
        logger.error("Ticket classification failed: %s", str(exc), exc_info=True)
        return {
            "category": "unknown",
            "urgency": "unknown",
            "sentiment": "unknown",
        }


async def retrieve_doc_step(state: TicketState, config: RunnableConfig) -> dict:
    """Step 2: Retrieve a relevant doc snippet from /documents/index.txt sources."""
    documents_dir = Path(__file__).parent / "documents"
    index_path = documents_dir / "index.txt"
    if not index_path.exists():
        logger.info("doc index missing: %s", str(index_path))
        return {"doc_snippet": "No doc snippet available."}

    sources = [line.strip() for line in index_path.read_text().splitlines() if line.strip()]
    if not sources:
        logger.info("doc index empty: %s", str(index_path))
        return {"doc_snippet": "No doc snippet available."}

    docs: list[dict[str, str]] = []
    for source in sources:
        try:
            if source.startswith(("http://", "https://")):
                logger.info("source: %s", source)
                text = await asyncio.to_thread(fetch_url_text, source)
            else:
                documents_path = (Path(__file__).parent / "documents" / source).resolve()
                path = str(Path(__file__).parent) + "/documents" + source
                if not Path(path).exists():
                    logger.info("doc file missing: %s", str(path))
                    continue
                text = read_local_text(Path(path))
            if text:
                docs.append({"source": source, "text": text})
        except Exception as exc:
            logger.error("Failed loading doc source %s: %s", source, str(exc), exc_info=True)

    if not docs:
        return {"doc_snippet": "No doc snippet available."}

    chunks: list[dict[str, str]] = []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": chunk})

    ticket_tokens = tokenize(state["content"])
    doc_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    idf = compute_idf(doc_tokens)
    query_vec = tfidf_vector(ticket_tokens, idf)

    best_score = 0.0
    best_chunk: dict[str, str] | None = None
    for chunk, tokens in zip(chunks, doc_tokens):
        chunk_vec = tfidf_vector(tokens, idf)
        score = cosine_similarity(query_vec, chunk_vec)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if not best_chunk or best_score == 0.0:
        return {"doc_snippet": "No doc snippet available."}

    snippet = extract_snippet(best_chunk["text"])
    return {"doc_snippet": f"Source: {best_chunk['source']}\n{snippet}"}


async def summarize_ticket_step(state: TicketState, config: RunnableConfig) -> dict:
    """Step 3: Summarize the ticket using the retrieved doc snippet."""
    try:
        logger.info("Summarizing support ticket")
        message = [
            {
                "role": "user",
                "content": (
                    "You are a support summarizer. Write a concise 1-2 sentence summary of the issue.\n"
                    "If the doc snippet is relevant, incorporate the key guidance in the summary.\n"
                    "Do not invent doc content.\n\n"
                    "IMPORTANT: You MUST return ONLY valid JSON in this exact format:\n"
                    '{"summary": "your summary text here"}\n'
                    "Do not include any other text, explanations, or markdown formatting.\n\n"
                    f"doc_snippet:\n{state['doc_snippet']}\n\n"
                    f"Category: {state['category']}\n"
                    f"Urgency: {state['urgency']}\n"
                    f"Sentiment: {state['sentiment']}\n\n"
                    f"Ticket:\n{state['content']}"
                ),
            }
        ]
        response = await llm_support.ainvoke(message)
        return {
            "summary": response.content,
        }
    except Exception as exc:
        logger.error("Ticket summary failed: %s", str(exc), exc_info=True)
        return {
            "summary": "Unable to summarize the ticket.",
        }


async def decide_action_step(state: TicketState, config: RunnableConfig) -> dict:
    """Step 4: Decide the next action for the ticket."""
    try:
        logger.info("Deciding action for support ticket")
        message = [
            {
                "role": "user",
                "content": (
                    "You are a support decision agent. Choose the next action:\n"
                    "- respond: clear issue with guidance available\n"
                    "- escalate: complex, urgent, or unclear issue\n\n"
                    "Provide a brief reason tied to the summary, urgency, and sentiment.\n\n"
                    "IMPORTANT: You MUST return ONLY valid JSON in this exact format:\n"
                    '{"action": "respond", "reason": "your reason here"}\n'
                    "Do not include any other text, explanations, or markdown formatting.\n\n"
                    f"Summary: {state['summary']}\n"
                    f"Urgency: {state['urgency']}\n"
                    f"Sentiment: {state['sentiment']}\n"
                    f"doc_snippet: {state['doc_snippet']}"
                ),
            }
        ]
        response = await llm_support.ainvoke(message)
        data = json.loads(response.content)
        return {
            "action": data["action"],
            "reason": data["reason"],
        }
    except Exception as exc:
        logger.error("Action decision failed: %s", str(exc), exc_info=True)
        return {
            "action": "escalate",
            "reason": "Automated decision failed; escalate for manual review.",
        }


async def response_step(state: TicketState, config: RunnableConfig) -> dict:
    """Step 5: Generate a customer response."""
    try:
        logger.info("Generating support response")
        message = [
            {
                "role": "user",
                "content": (
                    "You are a support response writer. Draft a concise, professional, empathetic reply.\n"
                    "If the doc snippet is relevant, include the key steps or guidance.\n"
                    "Do not mention internal fields like category/urgency/sentiment.\n\n"
                    "IMPORTANT: You MUST return ONLY valid JSON in this exact format:\n"
                    '{"response": "your response text here"}\n'
                    "Do not include any other text, explanations, or markdown formatting.\n\n"
                    f"Ticket:\n{state['content']}\n\n"
                    f"Summary: {state['summary']}\n"
                    f"Action: {state['action']}\n"
                    f"doc_snippet: {state['doc_snippet']}"
                ),
            }
        ]
        response = await llm_support.ainvoke(message)
        data = json.loads(response.content)
        logger.info("response: %s", response)
        return {
            "response": data["response"],
        }
    except Exception as exc:
        logger.error("response failed: %s", str(exc), exc_info=True)
        return {
            "response": (
                "Thanks for reaching out. We are reviewing your request and will follow up shortly."
            ),
        }


workflow = StateGraph(TicketState)
workflow.add_node("classify_ticket", classify_ticket_step)
workflow.add_node("retrieve_doc", retrieve_doc_step)
workflow.add_node("summarize_ticket", summarize_ticket_step)
workflow.add_node("decide_action", decide_action_step)
workflow.add_node("response", response_step)

workflow.add_edge(START, "classify_ticket")
workflow.add_edge("classify_ticket", "retrieve_doc")
workflow.add_edge("retrieve_doc", "summarize_ticket")
workflow.add_edge("summarize_ticket", "decide_action")
workflow.add_edge("decide_action", "response")
workflow.add_edge("response", END)

support_ticket_agent = workflow.compile()