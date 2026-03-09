import json
import os
import re
import boto3
import urllib.request
import urllib.parse

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
KNOWLEDGE_BUCKET = os.environ.get("KNOWLEDGE_BUCKET", "wyn-associates")
KNOWLEDGE_PREFIX = os.environ.get("KNOWLEDGE_PREFIX", "knowledge/")

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
s3 = boto3.client("s3", region_name=BEDROCK_REGION)

# Cache knowledge chunks in memory (Lambda container reuse)
_knowledge_cache = {}


def _load_knowledge():
    """Load all knowledge chunks from S3 into memory (cached across invocations)."""
    if _knowledge_cache:
        return _knowledge_cache

    try:
        resp = s3.list_objects_v2(Bucket=KNOWLEDGE_BUCKET, Prefix=KNOWLEDGE_PREFIX)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            body = s3.get_object(Bucket=KNOWLEDGE_BUCKET, Key=key)["Body"].read()
            chunk = json.loads(body)
            name = key.split("/")[-1].replace(".json", "")
            _knowledge_cache[name] = chunk
    except Exception as e:
        print(f"Error loading knowledge: {e}")

    return _knowledge_cache


def _retrieve_chunks(query, max_chunks=3):
    """Retrieve the most relevant knowledge chunks based on keyword overlap."""
    chunks = _load_knowledge()
    if not chunks:
        return []

    query_lower = query.lower()
    query_words = set(re.findall(r'[a-z0-9]+', query_lower))

    scored = []
    for name, chunk in chunks.items():
        keywords = [k.lower() for k in chunk.get("keywords", [])]
        # Score: count matching keywords + partial matches in query
        score = 0
        for kw in keywords:
            if kw in query_lower:
                score += 3  # exact substring match in query
            elif any(w == kw for w in query_words):
                score += 2  # exact word match
            elif any(kw.startswith(w) or w.startswith(kw) for w in query_words if len(w) > 2):
                score += 1  # partial match

        # Also check if the topic name matches
        if chunk.get("topic", "").lower() in query_lower:
            score += 2
        if chunk.get("tab", "").lower() in query_lower:
            score += 1

        if score > 0:
            scored.append((score, name, chunk))

    # Sort by score descending, return top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[2] for item in scored[:max_chunks]]


def _ddg_search(query, max_results=3):
    """Search DuckDuckGo Instant Answer API for supplemental context."""
    try:
        encoded = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
        url = f"https://api.duckduckgo.com/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "WYNAssistant/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        for topic in (data.get("RelatedTopics") or [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])
        return " | ".join(results[:max_results]) if results else ""
    except Exception:
        return ""


# Lean system prompt: only navigation rules + spotlight targets (no content)
SYSTEM_PROMPT = """You are WYN Assistant, the AI chatbot for Yiqiao Yin's personal homepage at y-yin.io. You help visitors navigate the site and answer questions using the KNOWLEDGE CONTEXT provided below.

CRITICAL OUTPUT RULE: Your ENTIRE response must be ONLY a single JSON object. No preamble, no explanation, no text before or after the JSON. Output ONLY this:
{"reply": "your message here", "actions": []}

The "actions" array can contain objects with these types:
- {"type": "navigate", "tab": "<tab_name>"} — switches to a main tab
- {"type": "navigate_subtab", "tab": "watchlist", "subtab": "<subtab_name>"} — switches to a sub-tab within Watchlist
- {"type": "switch_ai_view", "view": "graph"|"list"} — switches the AI Watchlist between 3D Supply Chain graph and Ticker List views
- {"type": "spotlight", "target": "<data_spotlight_id>"} — highlights a specific element on the page

## Main Tabs (use with "navigate" action):
home, market, watchlist, portfolio, letters, research, teaching

## Navbar spotlight targets:
nav-home, nav-market, nav-watchlist, nav-portfolio, nav-letters, nav-research, nav-teaching

## Spotlight targets by tab:
Home: home-title, home-background, home-passion-project, home-deployed-apps, home-hf-apps-link, home-youtube-link, home-linkedin-link
Market: market-title, market-heatmaps, market-stock-heatmap, market-crypto-heatmap
Watchlist: watchlist-title, watchlist-sub-navbar, watchlist-stocks-btn, watchlist-ai-btn, watchlist-stocks-content, watchlist-ai-content, watchlist-stocks-grid, watchlist-stock-screener, watchlist-ai-view-toggle, watchlist-ai-3d-btn, watchlist-ai-list-btn, watchlist-ai-3d-graph, watchlist-ai-energy, watchlist-ai-chips, watchlist-ai-infrastructure, watchlist-ai-models, watchlist-ai-application
Portfolio: portfolio-title, portfolio-tape, portfolio-chart
Letters: letters-title, letters-description, letters-table, letters-2025 through letters-2012
Research: research-title, research-industry-reports, research-papers, research-conferences, research-student-awards, research-books, research-economics, research-asset-pricing, research-trading, research-useful-resources, research-hedge-fund-filings
Teaching: teaching-title, teaching-grad-appointments, teaching-chicago-booth, teaching-pace, teaching-pace-courses, teaching-columbia, teaching-columbia-courses, teaching-precollege, teaching-ai4all, teaching-udemy-courses, teaching-packt-courses, teaching-software-engineer, teaching-textbooks, teaching-collected-notes

## Navigation rules:
- Sub-tab content is only visible when active. Always navigate_subtab BEFORE spotlighting content inside a sub-tab.
- To show 3D graph: navigate watchlist -> navigate_subtab "ai" -> switch_ai_view "graph" -> spotlight "watchlist-ai-3d-graph".
- To show a layer's tickers: navigate watchlist -> navigate_subtab "ai" -> switch_ai_view "list" -> spotlight the layer.
- Order actions: navigate first, then navigate_subtab, then switch_ai_view, then spotlight.

## Behavior:
1. ENTIRE output = ONLY the JSON object. No text outside JSON.
2. Use navigate/spotlight actions when user asks to see something.
3. Use KNOWLEDGE CONTEXT below to answer questions about the site.
4. If SEARCH CONTEXT is provided, use it for general knowledge questions not covered by the site.
5. Keep replies conversational, brief (1-3 sentences). Replies are spoken aloud via TTS, so keep them natural.
6. NEVER nest JSON in the "reply" field — plain text only.

Example: {"reply": "Yiqiao teaches at Pace University and Columbia. Let me show you.", "actions": [{"type": "navigate", "tab": "teaching"}, {"type": "spotlight", "target": "teaching-pace-courses"}]}
"""


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        messages = body.get("messages", [])

        if not messages:
            return _response(400, {"error": "No messages provided"})

        # Get the latest user message
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        # Retrieve relevant knowledge chunks from S3
        relevant_chunks = _retrieve_chunks(last_user_msg)
        knowledge_text = ""
        if relevant_chunks:
            parts = []
            for chunk in relevant_chunks:
                parts.append(f"[{chunk.get('topic', 'unknown')}] {chunk.get('content', '')}")
            knowledge_text = "\n".join(parts)

        # Search DuckDuckGo for supplemental context
        search_context = ""
        if last_user_msg:
            search_context = _ddg_search(last_user_msg)

        # Build the full system prompt
        system = SYSTEM_PROMPT
        if knowledge_text:
            system += f"\n\nKNOWLEDGE CONTEXT (from site content — use this to answer site-related questions):\n{knowledge_text}"
        if search_context:
            system += f"\n\nSEARCH CONTEXT (from DuckDuckGo — use for general knowledge questions):\n{search_context}"

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system,
            "messages": messages,
        }

        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

        result = json.loads(response["body"].read())
        assistant_text = result["content"][0]["text"]

        parsed = _extract_json(assistant_text)

        return _response(200, parsed)

    except Exception as e:
        print(f"Error: {e}")
        return _response(500, {"reply": "Sorry, something went wrong. Please try again.", "actions": []})


def _extract_json(text):
    """Extract a valid JSON object with 'reply' and 'actions' from model output."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "reply" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    brace_positions = [m.start() for m in re.finditer(r'\{', text)]
    for start in brace_positions:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "reply" in parsed:
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
                break

    return {"reply": text, "actions": []}


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,x-api-key",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(body),
    }
