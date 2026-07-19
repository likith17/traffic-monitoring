# LLM helpers: supports both the OpenAI-compatible /v1/chat/completions API
# (OpenAI, OpenRouter, Ollama, etc.) and the native Anthropic /v1/messages API.
# Which backend is used is controlled by the LLM_PROVIDER env var (default: openai).
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


def zone_from_lat(lat: float) -> str:
    """Map a latitude to a coarse Manhattan zone name.
    These thresholds roughly correspond to 96th St (upper boundary of Midtown)
    and 34th St (lower boundary of Midtown).
    """
    if lat > 40.78:
        return "Upper Manhattan"
    if lat > 40.75:
        return "Midtown"
    return "Lower Manhattan"


def build_traffic_context(
    merged: pd.DataFrame,
    segment_stats: pd.DataFrame | None = None,
    max_cameras_list: int = 25,
) -> str:
    """Serialise the current camera/segment data into a compact text block for the LLM.

    The output is intentionally plain text (not JSON) so it fits cleanly inside a
    system prompt without confusing models that treat JSON specially.
    max_cameras_list caps how many individual camera lines we include; the zonal
    summary is always included regardless.
    """
    df = merged.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["zone"] = df["lat"].apply(lambda x: zone_from_lat(float(x)) if pd.notna(x) else "Unknown")

    lines: list[str] = []
    lines.append("Manhattan DOT traffic cameras (congestion from the last batch merge).")
    lines.append(
        "Scores match camera_stats.csv at merge time; rerun update_camera_stats.py for newer frames."
    )

    if "score" in df.columns and df["score"].notna().any():
        # Only cameras that have been scored are useful to the model.
        sub = df[df["score"].notna()].copy()
        lines.append("\n### Summary by zone (cameras with scores)")
        for zone in ["Midtown", "Upper Manhattan", "Lower Manhattan"]:
            z = sub[sub["zone"] == zone]
            if z.empty:
                continue
            hi = (z["level"] == "high").sum()
            med = (z["level"] == "medium").sum()
            lo = (z["level"] == "low").sum()
            lines.append(
                f"- **{zone}**: n={len(z)}, mean_score={z['score'].mean():.2f}, "
                f"high={hi}, medium={med}, low={lo}"
            )

        lines.append("\n### Highest congestion cameras (top by score)")
        top = (
            sub.sort_values("score", ascending=False)
            .head(max_cameras_list)[
                ["name", "zone", "score", "level", "vehicles", "pedestrians", "signals"]
            ]
        )
        for _, row in top.iterrows():
            lines.append(
                f"- {row['name']} [{row['zone']}]: score={row['score']:.1f}, level={row['level']}, "
                f"vehicles={row['vehicles']}, pedestrians={row['pedestrians']}, signals={row['signals']}"
            )
    else:
        lines.append("\n(No scores in merged data — run update_camera_stats.py first.)")
        lines.append(f"Total cameras listed: {len(df)}.")

    # Append offline segment data if available — useful context even when camera
    # scores are stale, because segments are computed from saved video clips.
    if segment_stats is not None and not segment_stats.empty:
        lines.append("\n### Offline video segments (if present)")
        cols = [c for c in ["name", "avg_score", "level"] if c in segment_stats.columns]
        if cols:
            for _, row in segment_stats[cols].iterrows():
                lines.append("- " + ", ".join(f"{c}={row[c]}" for c in cols))

    return "\n".join(lines)


def _anthropic_split_messages(
    messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """The Anthropic API takes the system prompt as a separate top-level field,
    not as a message with role="system".  This function splits the list apart.
    Multiple system messages (if any) are joined with double newlines.
    """
    system_chunks: list[str] = []
    rest: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            system_chunks.append(content)
        elif role in ("user", "assistant"):
            rest.append({"role": role, "content": content})
    return "\n\n".join(system_chunks), rest


def _chat_anthropic(
    messages: list[dict[str, str]],
    *,
    timeout: int,
) -> tuple[str | None, str | None]:
    """Call the native Anthropic Messages API.

    Environment variables:
      ANTHROPIC_API_KEY   — required
      ANTHROPIC_MODEL     — default: claude-haiku-4-5
      ANTHROPIC_MAX_TOKENS — default: 2048
      ANTHROPIC_API_VERSION — default: 2023-06-01
    Returns (reply_text, None) on success or (None, error_string) on failure.
    """
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    model = (os.environ.get("ANTHROPIC_MODEL") or "claude-haiku-4-5").strip()
    max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS") or "2048")
    api_version = (os.environ.get("ANTHROPIC_API_VERSION") or "2023-06-01").strip()

    if not api_key:
        return None, (
            "Need ANTHROPIC_API_KEY when LLM_PROVIDER=anthropic. "
            "Or use OpenRouter etc. with OPENAI_API_KEY + OPENAI_BASE_URL."
        )

    system, msgs = _anthropic_split_messages(messages)
    if not msgs:
        return None, "No user/assistant messages to send to Anthropic."

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": api_version,
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": msgs,
        "temperature": 0.35,
    }
    if system:
        body["system"] = system  # only include if non-empty; Anthropic rejects an empty string

    try:
        r = requests.post(url, json=body, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return None, f"Request failed: {e}"

    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:500]
        hint = ""
        if r.status_code == 404:
            # A 404 usually means the model ID doesn't exist in this API version.
            hint = (
                " If the error mentions model, set ANTHROPIC_MODEL to a current id from Anthropic's docs."
            )
        return None, f"Anthropic API error {r.status_code}: {detail}{hint}"

    try:
        data = r.json()
        blocks = data["content"]
        # The response is a list of typed content blocks; we want the first text block.
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text", "")).strip(), None
        return None, f"Unexpected Anthropic response shape: {data!r:.800}"
    except (KeyError, IndexError, TypeError) as e:
        return None, f"Unexpected Anthropic response: {e}"


def _chat_openai_compatible(
    messages: list[dict[str, str]],
    *,
    timeout: int,
) -> tuple[str | None, str | None]:
    """Call any OpenAI-compatible /v1/chat/completions endpoint.

    Works with OpenAI, OpenRouter, Ollama (set OPENAI_BASE_URL=http://localhost:11434/v1
    and any non-empty OPENAI_API_KEY), and similar providers.

    Environment variables:
      OPENAI_API_KEY   — required
      OPENAI_BASE_URL  — default: https://api.openai.com/v1
      OPENAI_MODEL     — default: gpt-4o-mini
    Returns (reply_text, None) on success or (None, error_string) on failure.
    """
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

    if not api_key:
        return None, (
            "Set OPENAI_API_KEY (OpenAI, OpenRouter, ...). Ollama: OPENAI_BASE_URL=http://localhost:11434/v1 "
            "and a dummy key. Claude direct: LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY."
        )

    url = f"{base}/chat/completions"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.35,
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return None, f"Request failed: {e}"

    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:500]
        return None, f"API error {r.status_code}: {detail}"

    try:
        data = r.json()
        choice = data["choices"][0]["message"]["content"]
        return str(choice).strip(), None
    except (KeyError, IndexError, TypeError) as e:
        return None, f"Unexpected API response: {e}"


def chat_completion(
    messages: list[dict[str, str]],
    *,
    timeout: int = 90,
) -> tuple[str | None, str | None]:
    """Route the request to the correct LLM backend based on LLM_PROVIDER.

    Set LLM_PROVIDER=anthropic (or "claude") to use the Anthropic API.
    Any other value (or omitting the variable) routes to the OpenAI-compatible path.
    Returns (reply_text, None) on success or (None, error_string) on failure.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if provider in ("anthropic", "claude"):
        return _chat_anthropic(messages, timeout=timeout)
    return _chat_openai_compatible(messages, timeout=timeout)

