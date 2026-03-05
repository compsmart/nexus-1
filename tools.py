"""
Nexus Agent Tools
-----------------
Generic personal-assistant tools the agent can invoke at runtime:
  - web_search    : DuckDuckGo search (no API key)
  - web_fetch     : Fetch + clean text from a URL
  - calculator    : Safe arithmetic / math expression evaluator
  - datetime_now  : Current date, time and timezone info
  - unit_convert  : Convert between common units (length, weight, temp, etc.)
  - timer_delta   : Days/hours/minutes between two dates
  - wikipedia     : Quick Wikipedia summary for a topic
  - python_exec   : Execute a Python code snippet in a subprocess sandbox
  - file_read     : Read a text file from disk
  - file_write    : Write / append text to a file
  - file_list     : List files and directories in a path
  - file_search   : Search for files by name pattern or content substring
  - ingest_document : Read a document and store its contents as searchable AMM facts
  - create_skill    : Create a draft markdown skill from structured input + AMM pointer
  - learn_skill     : Search the web for a topic and save a markdown draft skill + AMM pointer
  - list_skills     : List markdown skills from disk
  - show_skill      : Show structured excerpt for one skill
  - publish_skill   : Promote draft skill to published and update pointer metadata
"""

import logging
import math
import os
import re
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from config import AgentConfig
    from memory import AdaptiveModularMemory

from skills_store import ENV_PLACEHOLDER_RE, SkillStore, slugify_skill_id


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    tool_name: str
    output: str
    success: bool = True
    error: Optional[str] = None

    def to_context(self) -> str:
        if not self.success:
            return f"[{self.tool_name} ERROR] {self.error}"
        return self.output


# ---------------------------------------------------------------------------
# Web search (DuckDuckGo â€” no API key required)
# ---------------------------------------------------------------------------

class WebSearchTool:
    name = "web_search"
    description = (
        "Search the web for current information. "
        "Usage: [TOOL_CALL: web_search | your search query here]"
    )

    def run(self, query: str) -> ToolResult:
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(self.name, "", success=False,
                              error="duckduckgo-search not installed. Run: pip install duckduckgo-search")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query.strip(), max_results=4))
            if not results:
                return ToolResult(self.name, "No results found for that query.")
            lines = []
            for r in results:
                title   = r.get("title", "").strip()
                snippet = r.get("body", "").strip()
                url     = r.get("href", "").strip()
                lines.append(f"â€¢ {title}\n  {snippet}\n  {url}")
            return ToolResult(self.name, "\n\n".join(lines))
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Web fetch (playwright primary, requests+BeautifulSoup fallback)
# ---------------------------------------------------------------------------

class WebFetchTool:
    name = "web_fetch"
    description = (
        "Fetch and read a web page, returning its main text content. "
        "Accepts a URL or a sentence containing a URL/domain. "
        "Usage: [TOOL_CALL: web_fetch | https://example.com]"
    )
    MAX_CHARS = 4000
    # Matches explicit https?:// URLs
    _FULL_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")
    # Matches bare domains like compsmart.cloud or www.example.com
    _DOMAIN_RE = re.compile(
        r"(?<!\w)(?:www\.)?([\w-]+\.(?:com|net|org|io|cloud|dev|ai|co|app"
        r"|site|uk|tech|edu|gov|info|me|us|nz|au|ca))(?:/[^\s]*)?(?!\w)",
        re.IGNORECASE,
    )
    _NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form"]

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract the first URL or domain from an arbitrary string."""
        # 1. Explicit https?:// URL
        m = self._FULL_URL_RE.search(text)
        if m:
            return m.group(0).rstrip(".,:;!?")
        # 2. Bare domain — prepend https://
        m = self._DOMAIN_RE.search(text)
        if m:
            return "https://" + m.group(0).lstrip("/").rstrip(".,:;!?")
        return None

    def _clean_html(self, html: str) -> str:
        """Strip noise tags and return plain text, truncated."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Very basic fallback if bs4 missing
            text = re.sub(r"<[^>]+>", " ", html)
            return " ".join(text.split())[: self.MAX_CHARS]
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(self._NOISE_TAGS):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        return "\n".join(lines)[: self.MAX_CHARS]

    def _fetch_playwright(self, url: str) -> str:
        """Render the page with a headless Chromium browser (JS support)."""
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/122.0.0.0 Safari/537.36"
            )
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)  # let JS paint
            html = page.content()
            browser.close()
        return self._clean_html(html)

    def _fetch_requests(self, url: str) -> str:
        """Fetch with requests + BeautifulSoup (static pages)."""
        import requests
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        resp.raise_for_status()
        return self._clean_html(resp.text)

    def run(self, raw: str) -> ToolResult:
        url = self._extract_url(raw.strip())
        if not url:
            return ToolResult(self.name, "", success=False,
                              error=f"No URL or domain found in: {raw[:120]}")

        # Try playwright first (handles JS/SPA sites), fall back to requests.
        errors = []
        try:
            text = self._fetch_playwright(url)
            if text:
                return ToolResult(self.name, f"[{url}]\n{text}")
        except ImportError:
            pass  # playwright not installed — go straight to requests
        except Exception as e:
            errors.append(f"playwright: {e}")
            logging.debug("web_fetch playwright failed for %s: %s", url, e)

        try:
            text = self._fetch_requests(url)
            if text:
                return ToolResult(self.name, f"[{url}]\n{text}")
            return ToolResult(self.name, "Page fetched but no readable text found.")
        except Exception as e:
            errors.append(f"requests: {e}")
            return ToolResult(self.name, "", success=False,
                              error=" | ".join(errors))

# ---------------------------------------------------------------------------
# Python code execution (subprocess sandbox â€” isolated from agent process)
# ---------------------------------------------------------------------------

class PythonExecTool:
    name = "python_exec"
    description = (
        "Execute a Python code snippet and return its stdout/stderr output. "
        "Runs in a sandboxed subprocess with a 10-second timeout. "
        "Usage: [TOOL_CALL: python_exec | print(2 ** 32)]"
    )
    TIMEOUT = 10
    MAX_OUTPUT = 3000

    def run(self, code: str) -> ToolResult:
        code = code.strip()
        if not code:
            return ToolResult(self.name, "", success=False, error="No code provided.")
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=self.TIMEOUT,
            )
            os.unlink(tmp_path)
            output = result.stdout + result.stderr
            output = output[: self.MAX_OUTPUT]
            if not output.strip():
                output = "(no output)"
            return ToolResult(self.name, output)
        except subprocess.TimeoutExpired:
            return ToolResult(self.name, "", success=False,
                              error=f"Execution timed out after {self.TIMEOUT}s.")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# ---------------------------------------------------------------------------
# File read
# ---------------------------------------------------------------------------

class FileReadTool:
    name = "file_read"
    description = (
        "Read a text file from disk and return its contents (up to 4000 chars). "
        "Usage: [TOOL_CALL: file_read | C:/path/to/file.txt]"
    )
    MAX_CHARS = 4000

    def run(self, path: str) -> ToolResult:
        path = path.strip().strip('"\'')
        p = Path(path)
        if not p.exists():
            return ToolResult(self.name, "", success=False,
                              error=f"File not found: {path}")
        if not p.is_file():
            return ToolResult(self.name, "", success=False,
                              error=f"Path is not a file: {path}")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            truncated = text[: self.MAX_CHARS]
            suffix = f"\nâ€¦[truncated, {len(text)} chars total]" if len(text) > self.MAX_CHARS else ""
            return ToolResult(self.name, truncated + suffix)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# ---------------------------------------------------------------------------
# File write
# ---------------------------------------------------------------------------

class FileWriteTool:
    name = "file_write"
    description = (
        "Write or append text to a file. "
        "Usage: [TOOL_CALL: file_write | path=/tmp/out.txt mode=write\nYour content here] "
        "mode can be 'write' (overwrite) or 'append'. Defaults to 'write'."
    )

    _HEADER_RE = re.compile(
        r'^path=(.+?)\s+mode=(write|append)\s*\n',
        re.IGNORECASE,
    )

    def run(self, arg: str) -> ToolResult:
        m = self._HEADER_RE.match(arg)
        if m:
            path    = m.group(1).strip().strip('"\'')
            mode    = "a" if m.group(2).lower() == "append" else "w"
            content = arg[m.end():]
        else:
            # Fallback: first line = path, rest = content
            lines   = arg.split("\n", 1)
            path    = lines[0].strip().strip('"\'')
            content = lines[1] if len(lines) > 1 else ""
            mode    = "w"
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.open(mode, encoding="utf-8").write(content)
            action = "Appended" if mode == "a" else "Written"
            return ToolResult(self.name,
                              f"{action} {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# ---------------------------------------------------------------------------
# File list
# ---------------------------------------------------------------------------

class FileListTool:
    name = "file_list"
    description = (
        "List files and directories at a given path. "
        "Usage: [TOOL_CALL: file_list | C:/Users/Brad/Documents]"
    )
    MAX_ENTRIES = 60

    def run(self, path: str) -> ToolResult:
        path = path.strip().strip('"\'')
        p = Path(path) if path else Path.cwd()
        if not p.exists():
            return ToolResult(self.name, "", success=False,
                              error=f"Path not found: {path}")
        try:
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            lines = []
            for entry in entries[: self.MAX_ENTRIES]:
                kind = "FILE" if entry.is_file() else "DIR "
                size = f"  {entry.stat().st_size:>10} B" if entry.is_file() else ""
                lines.append(f"[{kind}] {entry.name}{size}")
            suffix = (f"\nâ€¦and {len(entries) - self.MAX_ENTRIES} more"
                      if len(entries) > self.MAX_ENTRIES else "")
            return ToolResult(self.name,
                              f"{p.resolve()}\n" + "\n".join(lines) + suffix)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# ---------------------------------------------------------------------------
# File search
# ---------------------------------------------------------------------------

class FileSearchTool:
    name = "file_search"
    description = (
        "Search for files by name pattern or content substring within a directory tree. "
        "Usage: [TOOL_CALL: file_search | root=C:/projects pattern=*.py]  (name search)  "
        "or: [TOOL_CALL: file_search | root=C:/projects content=def train]  (content search)"
    )
    MAX_RESULTS = 20
    MAX_CHARS = 200  # context snippet per content match

    _ARG_RE = re.compile(
        r'root=(.+?)(?:\s+(?:pattern=(.+?)|content=(.+?)))?$',
        re.IGNORECASE,
    )

    def run(self, arg: str) -> ToolResult:
        m = self._ARG_RE.match(arg.strip())
        if not m:
            return ToolResult(self.name, "", success=False,
                              error="Format: root=<dir> pattern=<glob>  or  root=<dir> content=<text>")
        root    = Path(m.group(1).strip().strip('"\'' ))
        pattern = (m.group(2) or "").strip()
        content = (m.group(3) or "").strip()

        if not root.exists():
            return ToolResult(self.name, "", success=False,
                              error=f"Root not found: {root}")
        try:
            results = []
            if pattern:
                for p in root.rglob(pattern):
                    results.append(str(p))
                    if len(results) >= self.MAX_RESULTS:
                        break
            elif content:
                for p in root.rglob("*"):
                    if not p.is_file():
                        continue
                    try:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                        idx = text.lower().find(content.lower())
                        if idx >= 0:
                            snippet = text[max(0, idx-40): idx+self.MAX_CHARS].replace("\n", " ")
                            results.append(f"{p}\n  â€¦{snippet}â€¦")
                            if len(results) >= self.MAX_RESULTS:
                                break
                    except Exception:
                        continue
            else:
                return ToolResult(self.name, "", success=False,
                                  error="Provide either pattern=<glob> or content=<text>.")

            if not results:
                return ToolResult(self.name, "No matches found.")
            suffix = (f"\nâ€¦showing first {self.MAX_RESULTS} results"
                      if len(results) == self.MAX_RESULTS else "")
            return ToolResult(self.name, "\n".join(results) + suffix)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Forget / delete memories
# ---------------------------------------------------------------------------

class ForgetTool:
    """
    Deletes memories from AMM that closely match a query.
    Use this to correct wrong facts that were accidentally stored.

    A high similarity threshold (default 0.75) is used so only very closely
    matching entries are removed.

    Usage:
        [TOOL_CALL: forget | user's dog is called Fido]
        [TOOL_CALL: forget | my dog name Fido]
    """

    name = "forget"
    description = (
        "Delete wrong or outdated memories from long-term memory. "
        "Use this to correct facts that were stored incorrectly. "
        "Usage: [TOOL_CALL: forget | user's dog is called Fido]"
    )

    def __init__(self, memory: "AdaptiveModularMemory") -> None:
        self._memory = memory

    def run(self, query: str) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult(self.name, "", success=False, error="Empty query.")
        deleted = self._memory.delete_matching(query, threshold=0.75)
        if deleted == 0:
            return ToolResult(
                self.name,
                f"No closely matching memories found for: '{query}'. Nothing deleted."
            )
        # Flush immediately so the deletion is persisted to disk
        self._memory.flush()
        return ToolResult(
            self.name,
            f"Deleted {deleted} memory entry/entries matching: '{query}'. "
            "The wrong information has been removed.",
        )


# ---------------------------------------------------------------------------
# Search AMM memory explicitly
# ---------------------------------------------------------------------------

class SearchMemoryTool:
    """
    Lets the LLM explicitly query the agent's AMM long-term memory with a
    custom search query.  Use this BEFORE concluding you don't know something
    â€” the automatic retrieval pipeline may have missed a relevant memory.

    Usage:
        [TOOL_CALL: search_memory | user's dog name]
        [TOOL_CALL: search_memory | what city does the user live in]
    """

    name = "search_memory"
    description = (
        "Search your own long-term AMM memory for a specific topic or fact. "
        "Use this BEFORE saying you don't know something â€” the automatic context "
        "may have missed a relevant memory. "
        "Usage: [TOOL_CALL: search_memory | user's dog name]"
    )

    MAX_RESULTS = 5
    THRESHOLD = 0.25   # slightly lower than default to cast a wider net

    def __init__(self, memory: "AdaptiveModularMemory") -> None:
        self._memory = memory

    def run(self, query: str) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult(self.name, "", success=False, error="Empty query.")
        results = self._memory.retrieve(
            query,
            top_k=self.MAX_RESULTS,
            threshold=self.THRESHOLD,
        )
        if not results:
            return ToolResult(self.name, f"No memories found matching: '{query}'")
        lines = []
        for text, meta, score in results:
            mtype = (meta or {}).get("type", "?")
            lines.append(f"[{mtype}, score={score:.2f}] {text}")
        return ToolResult(self.name, "\n".join(lines))


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

def build_tool_registry(
    memory: "AdaptiveModularMemory",
    skill_store: Optional[SkillStore] = None,
    config: Optional["AgentConfig"] = None,
) -> dict:
    """Returns {tool_name: tool_instance} for all available tools."""
    if skill_store is None:
        skill_store = SkillStore(
            root_dir=(config.skills_root_dir if config else "skills"),
            drafts_dir=(config.skills_drafts_dir if config else "skills/drafts"),
            published_dir=(config.skills_published_dir if config else "skills/published"),
            require_env_placeholders=(config.skills_require_env_placeholders if config else True),
        )
    tools = [
        SearchMemoryTool(memory),
        ForgetTool(memory),
        WebSearchTool(),
        WebFetchTool(),
        CalculatorTool(),
        DateTimeTool(),
        UnitConverterTool(),
        TimerDeltaTool(),
        WikipediaTool(),
        PythonExecTool(),
        FileReadTool(),
        FileWriteTool(),
        FileListTool(),
        FileSearchTool(),
        IngestDocumentTool(memory),
        CreateSkillTool(memory, skill_store, config),
        LearnSkillTool(memory, skill_store, config),
        ListSkillsTool(skill_store),
        ShowSkillTool(skill_store),
        PublishSkillTool(memory, skill_store, config),
    ]
    return {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# Calculator â€” safe expression evaluator (no eval on arbitrary code)
# ---------------------------------------------------------------------------

class CalculatorTool:
    name = "calculator"
    description = (
        "Evaluate a mathematical expression and return the result. "
        "Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e. "
        "Usage: [TOOL_CALL: calculator | 2 ** 10 + sqrt(144)]"
    )

    _SAFE_NAMES = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan,   "log": math.log, "log10": math.log10,
        "log2": math.log2, "abs": abs,      "round": round,
        "floor": math.floor, "ceil": math.ceil,
        "pi": math.pi, "e": math.e, "inf": math.inf,
    }

    def run(self, expr: str) -> ToolResult:
        expr = expr.strip()
        # Reject anything that looks like an import or attribute access
        if re.search(r"__|import|exec|eval|open|os\.|sys\.", expr):
            return ToolResult(self.name, "", success=False,
                              error="Unsafe expression rejected.")
        try:
            result = eval(expr, {"__builtins__": {}}, self._SAFE_NAMES)  # noqa: S307
            return ToolResult(self.name, f"{expr} = {result}")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# ---------------------------------------------------------------------------
# DateTime â€” current date/time in local and UTC
# ---------------------------------------------------------------------------

class DateTimeTool:
    name = "datetime_now"
    description = (
        "Return the current date, time, and UTC offset. "
        "Usage: [TOOL_CALL: datetime_now | ]"
    )

    def run(self, _arg: str = "") -> ToolResult:
        now_local = datetime.now()
        now_utc   = datetime.now(timezone.utc)
        output = (
            f"Local : {now_local.strftime('%A, %d %B %Y  %H:%M:%S')}\n"
            f"UTC   : {now_utc.strftime('%A, %d %B %Y  %H:%M:%S')} UTC"
        )
        return ToolResult(self.name, output)


# ---------------------------------------------------------------------------
# Unit converter
# ---------------------------------------------------------------------------

class UnitConverterTool:
    name = "unit_convert"
    description = (
        "Convert a value between units. "
        "Usage: [TOOL_CALL: unit_convert | 100 km to miles]  "
        "Supports: kmâ†”miles, kgâ†”lbs, cmâ†”inches, mâ†”feet, Câ†”Fâ†”K, "
        "litersâ†”gallons, kphâ†”mph."
    )

    # (from_unit, to_unit) -> lambda value: converted_value
    _CONVERSIONS: dict = {
        # distance
        ("km",      "miles"):  lambda v: v * 0.621371,
        ("miles",   "km"):     lambda v: v / 0.621371,
        ("m",       "feet"):   lambda v: v * 3.28084,
        ("feet",    "m"):      lambda v: v / 3.28084,
        ("cm",      "inches"): lambda v: v * 0.393701,
        ("inches",  "cm"):     lambda v: v / 0.393701,
        # mass
        ("kg",      "lbs"):    lambda v: v * 2.20462,
        ("lbs",     "kg"):     lambda v: v / 2.20462,
        ("g",       "oz"):     lambda v: v * 0.035274,
        ("oz",      "g"):      lambda v: v / 0.035274,
        # temperature
        ("c",       "f"):      lambda v: v * 9/5 + 32,
        ("f",       "c"):      lambda v: (v - 32) * 5/9,
        ("c",       "k"):      lambda v: v + 273.15,
        ("k",       "c"):      lambda v: v - 273.15,
        ("f",       "k"):      lambda v: (v - 32) * 5/9 + 273.15,
        ("k",       "f"):      lambda v: (v - 273.15) * 9/5 + 32,
        # volume
        ("liters",  "gallons"):lambda v: v * 0.264172,
        ("gallons", "liters"): lambda v: v / 0.264172,
        ("ml",      "floz"):   lambda v: v * 0.033814,
        ("floz",    "ml"):     lambda v: v / 0.033814,
        # speed
        ("kph",     "mph"):    lambda v: v * 0.621371,
        ("mph",     "kph"):    lambda v: v / 0.621371,
    }

    _PARSE_RE = re.compile(
        r"([\d.]+)\s*([a-zA-ZÂ°/]+)\s+(?:to|in|->)\s+([a-zA-ZÂ°/]+)",
        re.IGNORECASE,
    )

    def run(self, arg: str) -> ToolResult:
        m = self._PARSE_RE.search(arg.strip())
        if not m:
            return ToolResult(self.name, "", success=False,
                              error="Format: '<value> <from_unit> to <to_unit>'. E.g. '5 km to miles'")
        value    = float(m.group(1))
        from_u   = m.group(2).lower().strip("Â°")
        to_u     = m.group(3).lower().strip("Â°")
        fn = self._CONVERSIONS.get((from_u, to_u))
        if fn is None:
            return ToolResult(self.name, "", success=False,
                              error=f"Unsupported conversion: {from_u} â†’ {to_u}")
        result = fn(value)
        return ToolResult(self.name, f"{value} {m.group(2)} = {result:.4g} {m.group(3)}")


# ---------------------------------------------------------------------------
# Timer / date delta
# ---------------------------------------------------------------------------

class TimerDeltaTool:
    name = "timer_delta"
    description = (
        "Calculate the time difference between two dates, or from today to a date. "
        "Usage: [TOOL_CALL: timer_delta | 2026-03-15]  (days until that date)  "
        "or: [TOOL_CALL: timer_delta | 2024-01-01 to 2026-02-26]"
    )

    _FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%B %d %Y", "%d %B %Y"]

    def _parse(self, s: str) -> Optional[datetime]:
        s = s.strip()
        for fmt in self._FORMATS:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    def run(self, arg: str) -> ToolResult:
        arg = arg.strip()
        # Two-date form
        sep = re.search(r"\bto\b", arg, re.IGNORECASE)
        if sep:
            d1 = self._parse(arg[: sep.start()])
            d2 = self._parse(arg[sep.end():])
        else:
            d1 = datetime.now()
            d2 = self._parse(arg)
        if d1 is None or d2 is None:
            return ToolResult(self.name, "", success=False,
                              error="Could not parse date(s). Use YYYY-MM-DD format.")
        delta = d2 - d1
        total_days = delta.days
        sign = "until" if total_days >= 0 else "since"
        abs_days = abs(total_days)
        years, rem = divmod(abs_days, 365)
        months, days = divmod(rem, 30)
        parts = []
        if years:  parts.append(f"{years} year{'s' if years != 1 else ''}")
        if months: parts.append(f"{months} month{'s' if months != 1 else ''}")
        if days or not parts: parts.append(f"{days} day{'s' if days != 1 else ''}")
        return ToolResult(self.name, f"{abs_days} days {sign} target ({', '.join(parts)})")


# ---------------------------------------------------------------------------
# Wikipedia summary
# ---------------------------------------------------------------------------

class WikipediaTool:
    name = "wikipedia"
    description = (
        "Fetch a short Wikipedia summary for a topic. "
        "Usage: [TOOL_CALL: wikipedia | Marie Curie]"
    )
    MAX_CHARS = 1200

    def run(self, topic: str) -> ToolResult:
        try:
            import requests
        except ImportError:
            return ToolResult(self.name, "", success=False, error="requests not installed.")
        topic = topic.strip()
        try:
            resp = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" +
                requests.utils.quote(topic),
                headers={"User-Agent": "NexusAgent/1.0"},
                timeout=8,
            )
            if resp.status_code == 404:
                return ToolResult(self.name, f"No Wikipedia article found for '{topic}'.")
            resp.raise_for_status()
            data = resp.json()
            extract = data.get("extract", "").strip()
            url     = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            if not extract:
                return ToolResult(self.name, "Article found but no extract available.")
            summary = extract[: self.MAX_CHARS]
            if len(extract) > self.MAX_CHARS:
                summary += "â€¦"
            return ToolResult(self.name, f"{summary}\n\nSource: {url}")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Document ingestion â€” loads a file into AMM as searchable fact chunks
# ---------------------------------------------------------------------------

class IngestDocumentTool:
    """
    Reads a text document (any .txt, .md, .csv, .py, .json, etc.) and splits
    it into overlapping chunks which are stored directly into the agent's AMM
    memory as typed "fact" entries.  After ingestion the agent can answer
    questions about the document using its normal retrieval pipeline.

    Usage:
        [TOOL_CALL: ingest_document | C:/path/to/file.txt]
        [TOOL_CALL: ingest_document | path=C:/path/to/file.md chunk_size=300]
    """

    name = "ingest_document"
    description = (
        "Load a document from disk and store its contents as searchable facts in memory "
        "so you can answer questions about it. "
        "Usage: [TOOL_CALL: ingest_document | /path/to/file.txt]  "
        "Optional: append  chunk_size=<int>  to control chunk length (default 400 words)."
    )

    # words per chunk; overlap between consecutive chunks
    DEFAULT_CHUNK_WORDS = 400
    OVERLAP_WORDS = 50
    MAX_CHUNKS = 200          # safety cap â€” prevents runaway ingestion
    MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB hard limit

    _ARG_RE = re.compile(
        r'^(?:path=)?(.+?)(?:\s+chunk_size=(\d+))?$',
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, memory: "AdaptiveModularMemory") -> None:
        self._memory = memory

    def run(self, arg: str) -> ToolResult:
        m = self._ARG_RE.match(arg.strip())
        if not m:
            return ToolResult(self.name, "", success=False,
                              error="Could not parse argument. Usage: /path/to/file.txt")

        file_path = Path(m.group(1).strip().strip("\"' "))
        chunk_words = int(m.group(2)) if m.group(2) else self.DEFAULT_CHUNK_WORDS

        if not file_path.exists():
            return ToolResult(self.name, "", success=False,
                              error=f"File not found: {file_path}")
        if not file_path.is_file():
            return ToolResult(self.name, "", success=False,
                              error=f"Path is not a file: {file_path}")
        if file_path.stat().st_size > self.MAX_FILE_BYTES:
            return ToolResult(self.name, "", success=False,
                              error=f"File too large (max 5 MB): {file_path.stat().st_size} bytes")

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(self.name, "", success=False,
                              error=f"Could not read file: {e}")

        words = text.split()
        if not words:
            return ToolResult(self.name, "File is empty â€” nothing to ingest.")

        # Build overlapping word-window chunks so facts that span a boundary
        # are still retrievable from either adjacent chunk.
        step = max(1, chunk_words - self.OVERLAP_WORDS)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start: start + chunk_words])
            chunks.append(chunk)
            if len(chunks) >= self.MAX_CHUNKS:
                break

        doc_name = file_path.name
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            self._memory.add_memory(
                f"[{doc_name} Â§{i + 1}/{total}] {chunk}",
                {
                    "type": "fact",
                    "subject": "document",
                    "source": str(file_path),
                    "chunk": i + 1,
                    "total_chunks": total,
                    "timestamp": time.time(),
                },
            )

        # Flush immediately so facts survive even if the agent is killed
        self._memory.flush()

        truncation_note = (
            f" (file truncated at {self.MAX_CHUNKS} chunks)" if total == self.MAX_CHUNKS else ""
        )
        return ToolResult(
            self.name,
            f"Ingested '{doc_name}' into AMM memory: "
            f"{total} chunk(s), ~{chunk_words} words each{truncation_note}. "
            "You can now answer questions about this document.",
        )


# ---------------------------------------------------------------------------
# Skill catalog tools (markdown-on-disk + AMM pointer index)
# ---------------------------------------------------------------------------


class CreateSkillTool:
    name = "create_skill"
    description = (
        "Create a draft skill directly from structured input and index it in AMM. "
        "Usage: [TOOL_CALL: create_skill | "
        "title=Image API Skill; skill_id=image-api-v1; summary=Generate images via API; "
        "endpoint=https://api.example.com/v1/images; tags=image,api; "
        "requires_env=IMG_API_KEY; capabilities=api_call,image_generation]"
    )

    def __init__(
        self,
        memory: "AdaptiveModularMemory",
        skill_store: SkillStore,
        config: Optional["AgentConfig"] = None,
    ) -> None:
        self._memory = memory
        self._skill_store = skill_store
        self._pointer_type = (
            (config.memory_skill_pointer_type if config else "skill_ref") or "skill_ref"
        )

    def _parse(self, arg: str) -> dict:
        raw = (arg or "").strip()
        if not raw:
            raise ValueError("Missing skill specification.")

        # JSON payload path first.
        if raw.startswith("{"):
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("JSON payload must be an object.")
            return payload

        # Fallback key-value parser: key=value; key2=value2
        data: dict = {}
        for token in [t.strip() for t in raw.split(";") if t.strip()]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            data[key.strip().lower()] = value.strip()
        if not data:
            raise ValueError("Invalid format. Provide JSON or semicolon-separated key=value fields.")
        return data

    def _to_list(self, data: dict, key: str) -> List[str]:
        value = data.get(key)
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in str(value).split(",") if v.strip()]

    def _build_sections(
        self,
        title: str,
        summary: str,
        endpoint: str,
        requires_env: List[str],
        request_format: str,
        example_request: str,
        example_response: str,
        steps: str,
        failure_modes: str,
    ) -> dict:
        env_lines = (
            "\n".join(f"- `${{{name}}}`" for name in requires_env)
            if requires_env
            else "- None required."
        )
        auth_var = requires_env[0] if requires_env else "API_KEY"
        safe_endpoint = endpoint or "https://api.example.com/v1/endpoint"
        req_fmt = request_format or "JSON over HTTPS with required auth headers."
        ex_req = example_request or (
            "```bash\n"
            f"curl -X POST \"{safe_endpoint}\" \\\n"
            f"  -H \"Authorization: Bearer ${{{auth_var}}}\" \\\n"
            "  -H \"Content-Type: application/json\" \\\n"
            "  -d '{\"prompt\":\"your prompt\"}'\n"
            "```"
        )
        ex_resp = example_response or (
            "```json\n"
            "{\"id\":\"example-id\",\"status\":\"ok\"}\n"
            "```"
        )
        step_text = steps or (
            "1. Validate required environment variables.\n"
            "2. Build request payload.\n"
            "3. Call endpoint.\n"
            "4. Validate response shape.\n"
            "5. Handle retries/errors."
        )
        fail_text = failure_modes or (
            "- Invalid credentials.\n"
            "- Rate limiting.\n"
            "- Invalid payload schema.\n"
            "- Transient network failure."
        )
        return {
            "Purpose": summary,
            "Preconditions": (
                "- Network access to the endpoint.\n"
                "- Correct API permissions and auth scope.\n"
                "- Request/response schema awareness."
            ),
            "Environment Variables": env_lines,
            "Endpoint": f"`POST {safe_endpoint}`",
            "Request Format": req_fmt,
            "Example Request": ex_req,
            "Example Response": ex_resp,
            "Step-by-Step Procedure": step_text,
            "Failure Modes": fail_text,
            "Validation Checklist": (
                "- [ ] Uses `${ENV_VAR}` placeholders only.\n"
                "- [ ] Request and response examples are valid.\n"
                "- [ ] Error handling is documented."
            ),
        }

    def run(self, arg: str) -> ToolResult:
        try:
            data = self._parse(arg)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))

        title = str(data.get("title", "")).strip()
        summary = str(data.get("summary", "")).strip()
        endpoint = str(data.get("endpoint", "")).strip()
        if not title:
            return ToolResult(self.name, "", success=False, error="Missing required field: title")
        if not summary:
            summary = f"Procedural skill: {title}"

        skill_id = str(data.get("skill_id", "")).strip() or f"{slugify_skill_id(title)}-v1"
        tags = self._to_list(data, "tags") or ["manual"]
        requires_env = self._to_list(data, "requires_env")
        capabilities = self._to_list(data, "capabilities") or ["knowledge_procedure"]
        source_urls = self._to_list(data, "source_urls")

        sections = self._build_sections(
            title=title,
            summary=summary,
            endpoint=endpoint,
            requires_env=requires_env,
            request_format=str(data.get("request_format", "")).strip(),
            example_request=str(data.get("example_request", "")).strip(),
            example_response=str(data.get("example_response", "")).strip(),
            steps=str(data.get("steps", "")).strip(),
            failure_modes=str(data.get("failure_modes", "")).strip(),
        )

        try:
            doc = self._skill_store.create_or_update_draft(
                skill_id=skill_id,
                title=title,
                summary=summary,
                source_urls=source_urls,
                tags=tags,
                requires_env=requires_env,
                capabilities=capabilities,
                sections=sections,
                owner="manual",
                trust_level="unverified",
            )
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=f"Failed to create draft skill: {e}")

        pointer_text = (
            f"[SKILL_REF] {title} | {skill_id} | {' '.join(summary.split())[:180]} | tags: {','.join(tags)}"
        )
        pointer_meta = {
            "type": self._pointer_type,
            "subject": skill_id,
            "skill_id": skill_id,
            "skill_path": str(doc.path),
            "status": "draft",
            "version": doc.version,
            "summary": summary,
            "title": title,
            "tags": tags,
            "timestamp": time.time(),
        }
        self._memory.upsert_by_meta(pointer_text, pointer_meta, match_keys=["type", "skill_id"])
        self._memory.flush()

        return ToolResult(
            self.name,
            f"Created draft skill '{skill_id}' at {doc.path}. "
            f"Indexed pointer in AMM. Publish with: [TOOL_CALL: publish_skill | {skill_id}]",
        )


class LearnSkillTool:
    name = "learn_skill"
    description = (
        "Search the web to learn a topic and save a draft markdown skill file. "
        "Stores a compact AMM pointer so the skill is retrievable without bloating memory. "
        "Usage: [TOOL_CALL: learn_skill | how to generate images with an API]"
    )

    MAX_SOURCES = 3
    MAX_CONTENT_CHARS = 5000

    def __init__(
        self,
        memory: "AdaptiveModularMemory",
        skill_store: SkillStore,
        config: Optional["AgentConfig"] = None,
    ) -> None:
        self._memory = memory
        self._skill_store = skill_store
        self._pointer_type = (
            (config.memory_skill_pointer_type if config else "skill_ref") or "skill_ref"
        )

    def _extract_tags(self, topic: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", topic.lower())
        stop = {"how", "to", "with", "for", "the", "and", "from", "using", "api"}
        tags = [t for t in tokens if t not in stop]
        return tags[:6] or ["general"]

    def _infer_capabilities(self, topic: str, corpus: str) -> List[str]:
        text = f"{topic} {corpus}".lower()
        caps = []
        if "api" in text or "endpoint" in text:
            caps.append("api_call")
        if "image" in text:
            caps.append("image_generation")
        if "json" in text:
            caps.append("json_request")
        if not caps:
            caps.append("knowledge_procedure")
        return caps

    def _detect_requires_env(self, corpus: str) -> List[str]:
        vars_found = set(ENV_PLACEHOLDER_RE.findall(corpus))
        keyish = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", corpus)
        for name in keyish:
            if any(k in name for k in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                vars_found.add(name)
        return sorted(vars_found)[:4]

    def _build_sections(
        self,
        topic: str,
        endpoint_url: str,
        summary: str,
        requires_env: List[str],
    ) -> dict:
        env_lines = (
            "\n".join(f"- `${{{name}}}`" for name in requires_env)
            if requires_env
            else "- None required by the captured references. Add placeholders when integrating credentials."
        )
        auth_var = requires_env[0] if requires_env else "API_KEY"
        endpoint = endpoint_url or "Provider-specific endpoint. Replace with your target API URL."
        return {
            "Purpose": summary,
            "Preconditions": (
                "- Access to the provider documentation.\n"
                "- Network connectivity and request tooling (curl, Python requests, or equivalent).\n"
                "- Proper authentication scope for the target endpoint."
            ),
            "Environment Variables": env_lines,
            "Endpoint": f"`POST {endpoint}`",
            "Request Format": (
                "JSON request over HTTPS with bearer authorization if required.\n"
                "Include prompt/body fields required by the provider."
            ),
            "Example Request": (
                "```bash\n"
                f"curl -X POST \"{endpoint}\" \\\n"
                f"  -H \"Authorization: Bearer ${{{auth_var}}}\" \\\n"
                "  -H \"Content-Type: application/json\" \\\n"
                "  -d '{\"prompt\":\"describe desired output\"}'\n"
                "```"
            ),
            "Example Response": (
                "```json\n"
                "{\"id\":\"example-id\",\"status\":\"ok\",\"result\":\"...\"}\n"
                "```"
            ),
            "Step-by-Step Procedure": (
                "1. Confirm endpoint and authentication requirements.\n"
                "2. Export required environment variables.\n"
                "3. Build a minimal valid JSON payload.\n"
                "4. Send a request and inspect status/body.\n"
                "5. Add retry/backoff and error handling.\n"
                "6. Validate output structure before downstream use."
            ),
            "Failure Modes": (
                "- Authentication failure (401/403).\n"
                "- Rate limiting (429).\n"
                "- Schema validation errors (400).\n"
                "- Timeout or transient network errors."
            ),
            "Validation Checklist": (
                "- [ ] Uses `${ENV_VAR}` placeholders only (no inline secrets).\n"
                "- [ ] Endpoint and payload shape are documented.\n"
                "- [ ] Example request/response are syntactically valid.\n"
                "- [ ] Failure handling and retries are defined."
            ),
        }

    def _build_pointer_text(self, title: str, skill_id: str, summary: str, tags: List[str]) -> str:
        tag_text = ",".join(tags)
        compact_summary = " ".join(summary.split())[:180]
        return f"[SKILL_REF] {title} | {skill_id} | {compact_summary} | tags: {tag_text}"

    def run(self, topic: str) -> ToolResult:
        topic = topic.strip()
        if not topic:
            return ToolResult(self.name, "", success=False, error="No topic provided.")

        try:
            from ddgs import DDGS
            import requests
            from bs4 import BeautifulSoup
        except ImportError as e:
            return ToolResult(self.name, "", success=False, error=f"Missing dependency: {e}")

        try:
            search_results = []
            with DDGS() as ddgs:
                for row in ddgs.text(topic, max_results=self.MAX_SOURCES + 2):
                    search_results.append(row)
                    if len(search_results) >= self.MAX_SOURCES + 2:
                        break
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=f"Search failed: {e}")

        if not search_results:
            return ToolResult(self.name, f"No search results found for: '{topic}'")

        source_rows = []
        for result in search_results[: self.MAX_SOURCES]:
            url = (result.get("href") or "").strip()
            snippet = (result.get("body") or "").strip()
            if not url:
                continue

            page_text = snippet
            try:
                resp = requests.get(url, timeout=10, headers={"User-Agent": "NexusAgent/1.0"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                page_text = soup.get_text(separator=" ", strip=True)[: self.MAX_CONTENT_CHARS]
            except Exception:
                pass

            if not page_text:
                continue
            source_rows.append({"url": url, "snippet": snippet, "text": page_text})

        if not source_rows:
            return ToolResult(self.name, "", success=False, error=f"No readable source content found for '{topic}'")

        corpus = " ".join(row["text"] for row in source_rows)
        summary = " ".join((source_rows[0]["snippet"] or corpus[:240]).split())[:260]
        skill_id = f"{slugify_skill_id(topic)}-v1"
        title = topic.strip().title()
        tags = self._extract_tags(topic)
        capabilities = self._infer_capabilities(topic, corpus)
        requires_env = self._detect_requires_env(corpus)
        endpoint_url = source_rows[0]["url"]
        sections = self._build_sections(topic, endpoint_url, summary, requires_env)

        try:
            skill_doc = self._skill_store.create_or_update_draft(
                skill_id=skill_id,
                title=title,
                summary=summary,
                source_urls=[row["url"] for row in source_rows],
                tags=tags,
                requires_env=requires_env,
                capabilities=capabilities,
                sections=sections,
                owner="auto",
                trust_level="unverified",
            )
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=f"Failed to write skill file: {e}")

        pointer_text = self._build_pointer_text(title, skill_id, summary, tags)
        pointer_meta = {
            "type": self._pointer_type,
            "subject": skill_id,
            "skill_id": skill_id,
            "skill_path": str(skill_doc.path),
            "status": "draft",
            "version": skill_doc.version,
            "summary": summary,
            "title": title,
            "tags": tags,
            "timestamp": time.time(),
        }
        self._memory.upsert_by_meta(
            pointer_text,
            pointer_meta,
            match_keys=["type", "skill_id"],
        )
        self._memory.flush()

        return ToolResult(
            self.name,
            f"Draft skill created for '{topic}' as '{skill_id}'. "
            f"Saved to {skill_doc.path}. "
            "A compact memory pointer was indexed. "
            f"Run [TOOL_CALL: publish_skill | {skill_id}] to promote it.",
        )


class ListSkillsTool:
    name = "list_skills"
    description = (
        "List skill files from the on-disk skill catalog. "
        "Usage: [TOOL_CALL: list_skills | ] or [TOOL_CALL: list_skills | published]"
    )

    def __init__(self, skill_store: SkillStore) -> None:
        self._skill_store = skill_store

    def run(self, arg: str) -> ToolResult:
        token = (arg or "").strip().lower()
        include_drafts = token in {"", "all", "draft", "drafts"}
        include_published = token in {"", "all", "published"}
        if token == "published":
            include_drafts = False
        if token in {"draft", "drafts"}:
            include_published = False

        docs = self._skill_store.list_skills(
            include_drafts=include_drafts,
            include_published=include_published,
        )
        if token and token not in {"all", "draft", "drafts", "published"}:
            docs = [
                d for d in docs
                if token in d.skill_id.lower()
                or token in d.title.lower()
                or token in ",".join(d.frontmatter.get("tags", [])).lower()
            ]

        if not docs:
            return ToolResult(self.name, "No skills found.")

        lines = []
        for doc in docs:
            tags = ",".join(doc.frontmatter.get("tags", []))
            lines.append(
                f"- {doc.skill_id} | {doc.title} | status={doc.status} | "
                f"v{doc.version} | tags={tags} | {doc.path}"
            )
        return ToolResult(self.name, "\n".join(lines))


class ShowSkillTool:
    name = "show_skill"
    description = (
        "Show a structured excerpt of a skill markdown file (draft or published). "
        "Usage: [TOOL_CALL: show_skill | image-gen-openai-v1]"
    )

    def __init__(self, skill_store: SkillStore) -> None:
        self._skill_store = skill_store

    def run(self, arg: str) -> ToolResult:
        skill_id = (arg or "").strip()
        if not skill_id:
            return ToolResult(self.name, "", success=False, error="Provide a skill_id.")
        doc = self._skill_store.load_skill(skill_id, include_drafts=True, include_published=True)
        if doc is None:
            return ToolResult(self.name, "", success=False, error=f"Skill not found: {skill_id}")
        excerpt = self._skill_store.render_excerpt(doc, max_chars=2800)
        return ToolResult(
            self.name,
            f"{doc.title} ({doc.skill_id}) status={doc.status} version={doc.version}\n"
            f"path={doc.path}\n\n{excerpt}",
        )


class PublishSkillTool:
    name = "publish_skill"
    description = (
        "Promote a draft skill to published status and refresh its AMM pointer metadata. "
        "Usage: [TOOL_CALL: publish_skill | image-gen-openai-v1]"
    )

    def __init__(
        self,
        memory: "AdaptiveModularMemory",
        skill_store: SkillStore,
        config: Optional["AgentConfig"] = None,
    ) -> None:
        self._memory = memory
        self._skill_store = skill_store
        self._pointer_type = (
            (config.memory_skill_pointer_type if config else "skill_ref") or "skill_ref"
        )

    def run(self, arg: str) -> ToolResult:
        skill_id = (arg or "").strip()
        if not skill_id:
            return ToolResult(self.name, "", success=False, error="Provide a skill_id.")
        try:
            doc = self._skill_store.publish_skill(skill_id)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))

        summary = (doc.frontmatter.get("summary") or doc.sections.get("Purpose", "")).strip()
        summary = " ".join(summary.split())[:180]
        tags = doc.frontmatter.get("tags", [])
        pointer_text = f"[SKILL_REF] {doc.title} | {doc.skill_id} | {summary} | tags: {','.join(tags)}"
        pointer_meta = {
            "type": self._pointer_type,
            "subject": doc.skill_id,
            "skill_id": doc.skill_id,
            "skill_path": str(doc.path),
            "status": "published",
            "version": doc.version,
            "summary": summary,
            "title": doc.title,
            "tags": tags,
            "timestamp": time.time(),
        }
        self._memory.upsert_by_meta(pointer_text, pointer_meta, match_keys=["type", "skill_id"])
        self._memory.flush()

        return ToolResult(
            self.name,
            f"Published skill '{doc.skill_id}' (v{doc.version}) at {doc.path}. "
            "AMM pointer metadata updated.",
        )



