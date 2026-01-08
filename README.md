# Support Agent

LangGraph-based support ticket agent that classifies, summarizes and responds to customer questions using supporting docs/URLs.

## Description

### Problem
In many companies, support teams handle a high volume of customer tickets that are largely unstructured, repetitive and dependent on internal knowledge bases that evolve frequently. Manually classifying issues, summarizing customer intent and crafting accurate responses is time consuming.

### Agentic Solution
This project implements a LangGraph based multi-agent support system that automates ticket classification, summarization and response generation using the provided documentation and URLs. Agents collaborate to interpret customer intent, retrieve relevant knowledge base content, generate responses. This system leverages agent level reasoning and self-evaluation instead of human annotation. As products and documentation evolve, the system adapts easily by simply updating the knowledge base.

## Deep Dive

The workflow runs in the following order:

1) `classify_ticket` (`classify_ticket_step`)
- Sends the ticket content to the LLM and asks for a strict JSON response with `category`, `urgency` and `sentiment`.
- Parses the JSON response and stores those fields in the state.

2) `retrieve_doc` (`retrieve_doc_step`)
- Loads sources listed in `support_agent/utils/documents/index.txt`.
- For URLs, it fetches text via `fetch_url_text`. For local files, it reads from `support_agent/utils/documents/`.
- Chunks documents with `chunk_text`, tokenizes with `tokenize`, and builds TF-IDF vectors.
- Computes cosine similarity between the ticket query vector and each chunk then selects the best match.
- Returns a `doc_snippet` with the matching source and extracted snippet text.

3) `summarize_ticket` (`summarize_ticket_step`)
- Sends the ticket content, doc snippet and classification fields to the LLM.
- Asks for a strict JSON summary which is stored in `summary`.

4) `decide_action` (`decide_action_step`)
- Uses the summary, urgency, sentiment, and doc snippet to choose `respond` or `escalate` (manual intervention).
- Returns a JSON `action` and `reason`.

5) `response` (`response_step`)
- Uses the ticket, summary and doc snippet to generate the final response.
- Returns a JSON `response` string which is what the CLI prints.

### Doc search approach

Doc search is implemented as a TF-IDF + cosine similarity retrieval:
- `tokenize` to build term lists per chunk.
- `compute_idf` and `tfidf_vector` to build vectors.
- `cosine_similarity` to score each chunk against the ticket query.
- `extract_snippet` to return a focused excerpt of the best match.
URL sources are scraped via Firecrawl for structured extraction.

![Project Architecture](/images/graph.png)

## Tech stack

- Python
- Poetry for dependency management
- LangGraph for workflow orchestration
- Hugging Face model for NLP
- LangSmith for cloud deployment and observability
- Firecrawl for structured URL scraping

## Setup

1. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies with Poetry:

```bash
poetry install
```


3. Create a `.env` file (or export env vars) with the following values:

- `HUGGINGFACEHUB_API_TOKEN`: access token for the Hugging Face model.
- `FIRECRAWL_API_KEY`: API key for Firecrawl URL scraping.
- `LANGSMITH_API_KEY`: LangSmith API key for tracing and observability.
- `LANGSMITH_TRACING`: set to `true` to enable LangSmith tracing.
- `LANGSMITH_PROJECT`: project name for grouping traces (example: `development`).

4. Update the knowledge base by adding local file paths or URLs to `support_agent/utils/documents/index.txt`.

5. Run locally

- Option 1: run the agent script and enter a question at the prompt:

```bash
python3 -m support_agent.agent
```

- Option 2: run LangGraph Studio for step-by-step execution (interactive UI):

```bash
poetry run langgraph dev --allow-blocking
```

Then open the LangGraph Studio page and send a message to the agent as a JSON or YAML object:

```json
{
    "content": "Your query"
}
```

## Sample Flow


The agent was provided with official Neon documentation URLs in `support_agent/utils/documents/index.txt` and a user query: *"What's the response time for my Neon ticket if the Severity level is 1?"* The agent parsed the docs and returned the following response based strictly on the documentation:

***Based on your support plan, for a Severity level 1 issue, you can expect a response within 1 hour for Production support plans or within 4 hours for Business support plans. Please refer to our response time guidelines for more information on [Link to Response times](https://neon.com/docs/introduction/support#response-times).***

![Project Flow](/images/flow1.png)

## Deployment to LangGraph Cloud

On push to files within `support-agent/**`, it triggers the `support_agent_cd` workflow. That workflow pushes code to the `main` branch in the specified repository, which then deploys to LangGraph Cloud.

The `support_agent_cd` workflow relies on a GitHub PAT token.

## Future enhancements

1) The current setup uses a free LLM. We can swap in a model that better fits our latency, cost and quality requirements.

2) For more relevant document extraction, use a hybrid retriever that combines BM25 with vector search.

3) Persist doc chunks and embeddings in a vector database so we do not re-embed on every query and can run hybrid retrieval against precomputed embeddings.

4) Expand structured scraping coverage and add more extraction rules in Firecrawl.

5) When the agent escalates to a human, nothing is queued as of now. We can add a SQL database to store escalated tickets so experts can pick them up in first-in order.
