"""Browser & Web Launch Skills for J.A.R.V.I.S. Phase 22.7.

Provides safe, minimal web navigation, web searching, and browser launching
utilizing strictly the Python standard-library `webbrowser.open()`.

Safety Invariants Enforced:
1. Allow ONLY URLs beginning with 'http://' or 'https://'.
2. Reject javascript:, file:, data:, vbscript:, about:, arbitrary custom schemes, UNC paths, and malformed URLs.
3. Reject empty, whitespace-only, or control-character-containing URLs.
4. Strictly NO subprocess, os.system, shell commands, PowerShell, cmd.exe, or shell interpolation.
5. All search queries are strictly escaped via urllib.parse.urlencode before URL generation.
6. Zero process termination, OS power operations, or host control capabilities.
7. Zero global mutable state; full dependency injection support.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union
import urllib.parse
import webbrowser

from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.security import (
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSecurityPolicy,
)

logger = get_logger("SYSTEM.BROWSER_SKILLS")

# Default landing page when opening default browser without a specific URL
DEFAULT_BROWSER_URL: Final[str] = "https://www.google.com"

# Default search engine identifier
DEFAULT_SEARCH_ENGINE: Final[str] = "google"

# Supported search engine HTTPS endpoints
SEARCH_ENGINE_ENDPOINTS: Final[Dict[str, str]] = {
    "google": "https://www.google.com/search",
    "bing": "https://www.bing.com/search",
    "duckduckgo": "https://duckduckgo.com/",
    "ddg": "https://duckduckgo.com/",
    "yahoo": "https://search.yahoo.com/search",
}

# Permitted URL schemes (strict allowlist)
ALLOWED_SCHEMES: Final[Set[str]] = {"http", "https"}

# Explicitly prohibited schemes for fast descriptive rejection
PROHIBITED_SCHEMES: Final[Set[str]] = {
    "javascript",
    "file",
    "data",
    "vbscript",
    "about",
    "chrome",
    "edge",
    "ms-settings",
    "blob",
    "ftp",
    "ws",
    "wss",
    "ssh",
    "telnet",
    "mailto",
    "ldap",
    "gopher",
}

# Regex pattern matchers for natural language intent routing
_RE_OPEN_URL = re.compile(
    r"^(?:open|browse|go\s+to|visit)\s+(?:url\s+|website\s+|link\s+|webpage\s+)?(\S+)$",
    re.IGNORECASE,
)
_RE_SEARCH_WEB = re.compile(
    r"^(?:search(?:\s+the)?\s+web\s+(?:for\s+)?|search\s+for\s+|google\s+)(.+)$",
    re.IGNORECASE,
)
_RE_OPEN_BROWSER = re.compile(
    r"^(?:open|launch|start)\s+(?:the\s+)?(?:default\s+)?browser$",
    re.IGNORECASE,
)


def is_url_or_domain(text: Optional[str]) -> bool:
    """Check if a target string represents a web URL or domain.

    Matches:
    - Full URLs starting with http:// or https://
    - Web domains starting with www.
    - Domain patterns with web TLDs (e.g. google.com, sub.domain.org, github.io)
    """
    if not text or not isinstance(text, str):
        return False
    clean = text.strip().lower()
    if clean.startswith(("http://", "https://", "www.")):
        return True
    domain_match = re.match(
        r"^[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}(?:/[^\s]*)?$", clean
    )
    if domain_match:
        host_part = clean.split("/")[0]
        ext = host_part.split(".")[-1].lower() if "." in host_part else ""
        if ext in ("exe", "bat", "cmd", "msi", "lnk"):
            return False
        return True
    return False


def validate_url(url: Any) -> str:
    """Validate that a URL starts strictly with http:// or https:// and is well-formed.

    Enforces:
    - Non-empty string requirement
    - Rejection of empty or whitespace-only strings
    - Rejection of null bytes and ASCII control characters
    - Rejection of UNC paths (\\\\server\\share or //server/share)
    - Rejection of unencoded whitespace within the URL
    - Rejection of dangerous schemes (javascript:, file:, data:, vbscript:, about:, etc.)
    - Rejection of non-http/https schemes (ftp:, custom:, etc.)
    - Presence of a valid host/netloc component

    Args:
        url: The candidate URL to validate.

    Returns:
        The validated, stripped URL string.

    Raises:
        SkillExecutionError: If URL is empty, non-string, unparseable, or malformed.
        SecurityPolicyViolationError: If URL uses a prohibited or unauthorized scheme, or is a UNC path.
    """
    if url is None or not isinstance(url, str):
        raise SkillExecutionError("URL must be a non-empty string.")

    clean_url = url.strip()
    if not clean_url:
        raise SkillExecutionError("URL cannot be empty or whitespace-only.")

    # Control characters and null bytes check
    if any(ord(c) < 32 or ord(c) == 127 for c in clean_url):
        raise SecurityPolicyViolationError("URL contains invalid control characters or null bytes.")

    # UNC paths check (e.g. \\server\share or //server/share)
    if clean_url.startswith(("\\\\", "//")):
        raise SecurityPolicyViolationError(
            f"UNC paths and network shares are prohibited: '{clean_url}'"
        )

    # Check for unencoded whitespace inside the URL
    if any(c.isspace() for c in clean_url):
        raise SkillExecutionError(
            f"Malformed URL: whitespace is not permitted within URLs: '{clean_url}'"
        )

    # Parse URL using standard library urlsplit
    try:
        parsed = urllib.parse.urlsplit(clean_url)
    except Exception as exc:
        raise SkillExecutionError(f"Malformed URL: unable to parse '{clean_url}': {exc}") from exc

    scheme = parsed.scheme.lower()

    # Scheme presence check
    if not scheme:
        raise SkillExecutionError(
            f"Malformed URL: missing scheme in '{clean_url}'. Only 'http://' and 'https://' are allowed."
        )

    # Scheme allowlist check
    if scheme in PROHIBITED_SCHEMES or scheme not in ALLOWED_SCHEMES:
        raise SecurityPolicyViolationError(
            f"Prohibited URL scheme '{scheme}'. Only 'http://' and 'https://' URLs are permitted."
        )

    # Host/netloc presence check (must not be empty, e.g. 'http://' alone or 'https:///path')
    if not parsed.netloc or not parsed.netloc.strip():
        raise SkillExecutionError(
            f"Malformed URL: missing domain/host in '{clean_url}'."
        )

    return clean_url


class BrowserSkills(BaseSystemSkill):
    """Production Browser & Web Launch Skills for J.A.R.V.I.S. Phase 22.7.

    Provides safe web launching:
    - open_url: Launch validated HTTP/HTTPS URLs via webbrowser.open()
    - search_web: Safe query-escaped HTTPS web search via standard search engines
    - open_browser: Open default browser home page safely
    """

    name: str = "browser"
    description: str = (
        "Safe browser navigation and web search launching via standard library webbrowser."
    )
    priority: int = 60
    tags: list[str] = ["browser", "web", "search", "url", "system", "windows"]
    permissions: set[str] = {"system:read", "system:execute"}

    def __init__(
        self,
        *,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, Any]] = None,
    ) -> None:
        """Initialize BrowserSkills instance."""
        super().__init__(
            name=self.name,
            description=self.description,
            priority=self.priority,
            tags=self.tags,
            permissions=self.permissions,
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            config=config,
            logger=logger or get_logger("SKILL.BROWSER"),
            container=container,
            event_bus=event_bus,
        )
        self._lock = threading.RLock()

    # -----------------------------------------------------------------------
    # Low-level Native Browser Launch Hook (Overridable / Mockable in Tests)
    # -----------------------------------------------------------------------

    def _webbrowser_open_hook(self, url: str) -> bool:
        """Invoke standard library webbrowser.open().

        Never executes shells, command prompts, or subprocesses.
        Overridable and mockable in unit test suites.
        """
        return bool(webbrowser.open(url))

    def _launch_browser(self, url: str) -> bool:
        """Safely invoke browser opening with error trapping and locking."""
        with self._lock:
            try:
                success = self._webbrowser_open_hook(url)
            except Exception as exc:
                self.logger.error("webbrowser.open failed for URL '%s': %s", url, exc)
                raise SkillExecutionError(f"Failed to launch browser: {exc}") from exc

            if not success:
                self.logger.warning("webbrowser.open returned False for URL '%s'", url)
                raise SkillExecutionError(f"Browser launch returned failure for URL: '{url}'")

            return True

    # -----------------------------------------------------------------------
    # BaseSystemSkill Protocol Methods
    # -----------------------------------------------------------------------

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can handle the given command."""
        op, target, _, _ = self.parse_command(command)
        if op in (
            "open_url",
            "browse_url",
            "open_link",
            "search_web",
            "web_search",
            "open_browser",
            "launch_browser",
            "start_browser",
        ):
            return True

        if isinstance(command, str):
            clean = command.strip()
            if _RE_OPEN_BROWSER.match(clean):
                return True
            if _RE_SEARCH_WEB.match(clean):
                return True
            m_url = _RE_OPEN_URL.match(clean)
            if m_url:
                target_cand = m_url.group(1).strip()
                if is_url_or_domain(target_cand):
                    return True

        return False

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Normalize arbitrary command inputs into structured components."""
        if isinstance(command, dict):
            return super().parse_command(command)

        text = str(command or "").strip()

        m_browser = _RE_OPEN_BROWSER.match(text)
        if m_browser:
            return "open_browser", None, {}, None

        m_search = _RE_SEARCH_WEB.match(text)
        if m_search:
            query_target = m_search.group(1).strip()
            return "search_web", query_target, {"query": query_target}, None

        m_url = _RE_OPEN_URL.match(text)
        if m_url:
            raw_target = m_url.group(1).strip()
            if is_url_or_domain(raw_target):
                if not raw_target.lower().startswith(("http://", "https://")):
                    url_target = f"https://{raw_target}"
                else:
                    url_target = raw_target
                return "open_url", url_target, {"url": url_target}, None

        return super().parse_command(command)

    def _execute_operation(
        self, operation: str, target: Optional[str], parameters: Dict[str, Any]
    ) -> Any:
        """Internal operation dispatcher for BrowserSkills."""
        op = operation.strip().lower()

        if op in ("open_url", "browse_url", "open_link"):
            raw_url = parameters.get("url") or target
            if not raw_url:
                raise SkillExecutionError("No URL specified for open_url operation.")
            url_str = str(raw_url).strip()
            if not url_str.lower().startswith(("http://", "https://")) and is_url_or_domain(url_str):
                url_str = f"https://{url_str}"
            return self.open_url(url_str)

        if op in ("search_web", "web_search"):
            raw_query = parameters.get("query") or parameters.get("q") or target
            if not raw_query:
                raise SkillExecutionError("No search query specified for search_web operation.")
            engine = parameters.get("engine") or DEFAULT_SEARCH_ENGINE
            return self.search_web(str(raw_query), engine=str(engine))

        if op in ("open_browser", "launch_browser", "start_browser"):
            raw_url = parameters.get("url") or target
            return self.open_browser(str(raw_url) if raw_url else None)

        raise SkillExecutionError(f"Unsupported browser operation '{operation}'.")

    # -----------------------------------------------------------------------
    # Public Concrete Operations
    # -----------------------------------------------------------------------

    def open_url(self, url: str) -> Dict[str, Any]:
        """Open a validated HTTP/HTTPS URL in the default web browser.

        Args:
            url: The HTTP or HTTPS URL to navigate to.

        Returns:
            Structured dictionary with keys: url (str) and opened (bool).

        Raises:
            SkillExecutionError: If URL is malformed or browser fails to launch.
            SecurityPolicyViolationError: If URL uses a prohibited scheme or UNC path.
        """
        validated = validate_url(url)
        self._launch_browser(validated)
        self.logger.info("Successfully opened URL: %s", validated)
        return {
            "url": validated,
            "opened": True,
        }

    def search_web(
        self,
        query: str,
        engine: str = DEFAULT_SEARCH_ENGINE,
    ) -> Dict[str, Any]:
        """Execute web search by constructing a safe HTTPS search URL and launching browser.

        Args:
            query: The search query text.
            engine: The search engine to use (default: 'google').

        Returns:
            Structured dictionary with keys: query, url, engine, opened.

        Raises:
            SkillExecutionError: If query is empty or browser fails to launch.
        """
        query_str = str(query or "").strip()
        if not query_str:
            raise SkillExecutionError("Search query cannot be empty or whitespace-only.")

        engine_key = str(engine or DEFAULT_SEARCH_ENGINE).strip().lower()
        endpoint = SEARCH_ENGINE_ENDPOINTS.get(
            engine_key, SEARCH_ENGINE_ENDPOINTS[DEFAULT_SEARCH_ENGINE]
        )

        # Safely encode query parameters without command concatenation or shell involvement
        encoded_query = urllib.parse.urlencode({"q": query_str})
        search_url = f"{endpoint}?{encoded_query}"

        # Validate the generated HTTPS URL before launching
        validated = validate_url(search_url)
        self._launch_browser(validated)
        self.logger.info("Successfully launched web search for query '%s' on %s", query_str, engine_key)

        return {
            "query": query_str,
            "url": validated,
            "engine": engine_key,
            "opened": True,
        }

    def open_browser(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Open the user's default browser or home page.

        Args:
            url: Optional starting URL. Defaults to DEFAULT_BROWSER_URL if omitted.

        Returns:
            Structured dictionary with keys: url, opened, default.

        Raises:
            SkillExecutionError: If URL is invalid or browser launch fails.
            SecurityPolicyViolationError: If custom URL violates security policy.
        """
        if url is not None and str(url).strip():
            target_url = validate_url(str(url).strip())
            is_default = False
        else:
            target_url = DEFAULT_BROWSER_URL
            is_default = True

        self._launch_browser(target_url)
        self.logger.info("Successfully opened browser at '%s' (default=%s)", target_url, is_default)

        return {
            "url": target_url,
            "opened": True,
            "default": is_default,
        }


__all__ = [
    "ALLOWED_SCHEMES",
    "BrowserSkills",
    "DEFAULT_BROWSER_URL",
    "DEFAULT_SEARCH_ENGINE",
    "PROHIBITED_SCHEMES",
    "SEARCH_ENGINE_ENDPOINTS",
    "validate_url",
]
