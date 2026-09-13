"""Comprehensive unit, security, and invariant tests for Phase 22.7 Browser Skills.

SAFETY REQUIREMENT:
All browser invocations (webbrowser.open) are strictly 100% mocked.
NO REAL BROWSER IS EVER OPENED.
Zero subprocesses, shell commands, or power transitions are executed.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import os
import unittest
from unittest.mock import MagicMock, call, patch
import urllib.parse

from app.ai.planner.events import (
    PlannerEvent,
    PlannerEventBus,
    SystemSkillCompleted,
    SystemSkillFailed,
    SystemSkillStarted,
)
from app.core.container import ServiceContainer
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import SystemSkillResult
from app.skills.system.browser_skills import (
    ALLOWED_SCHEMES,
    BrowserSkills,
    DEFAULT_BROWSER_URL,
    DEFAULT_SEARCH_ENGINE,
    PROHIBITED_SCHEMES,
    SEARCH_ENGINE_ENDPOINTS,
    validate_url,
)
from app.skills.system.security import (
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSafetyTier,
    SystemSecurityPolicy,
)


class TestBrowserSkills(unittest.TestCase):
    """Unit and security test suite for BrowserSkills."""

    def setUp(self) -> None:
        """Initialize test environment with isolated event bus, security policy, and mocked browser hook."""
        self.container = ServiceContainer()
        self.bus = PlannerEventBus()
        self.security_policy = SystemSecurityPolicy(event_bus=self.bus)
        self.confirmation_manager = SystemConfirmationManager(event_bus=self.bus)

        self.skill = BrowserSkills(
            security_policy=self.security_policy,
            confirmation_manager=self.confirmation_manager,
            container=self.container,
            event_bus=self.bus,
        )

        # Track emitted planner events
        self.events: list[PlannerEvent] = []
        self.bus.subscribe(PlannerEvent, lambda e: self.events.append(e))

        # Class/instance mock hook so NO real browser is ever opened
        self.mock_open = MagicMock(return_value=True)
        self.skill._webbrowser_open_hook = self.mock_open

    # -----------------------------------------------------------------------
    # 1. Routing & Intent Recognition Tests
    # -----------------------------------------------------------------------

    def test_01_can_handle_all_browser_operations(self) -> None:
        """Verify can_handle recognizes all browser operations and natural language phrasing."""
        valid_cmds = [
            {"operation": "open_url", "url": "https://example.com"},
            {"operation": "browse_url", "url": "https://example.com"},
            {"operation": "open_link", "url": "https://example.com"},
            {"operation": "search_web", "query": "Python 3.13"},
            {"operation": "web_search", "query": "Python 3.13"},
            {"operation": "open_browser"},
            {"operation": "launch_browser"},
            {"operation": "start_browser"},
            "open url https://example.com",
            "browse https://example.com",
            "open google.com",
            "open https://example.com",
            "open www.python.org",
            "visit github.com/openai",
            "browse sub.domain.org/path",
            "go to website https://python.org",
            "visit link https://docs.python.org",
            "search web for quantum computing",
            "search the web for fast algorithms",
            "search for artificial intelligence",
            "google machine learning",
            "open browser",
            "launch browser",
            "start default browser",
        ]
        for cmd in valid_cmds:
            with self.subTest(cmd=cmd):
                self.assertTrue(self.skill.can_handle(cmd))

    def test_02_can_handle_rejects_unrelated_commands(self) -> None:
        """Verify can_handle strictly ignores unrelated actions."""
        invalid_cmds = [
            {"operation": "open_app", "target": "notepad"},
            {"operation": "delete_file", "path": "c:/test.txt"},
            {"operation": "set_volume", "level": 50},
            {"operation": "close_window", "target": "chrome"},
            "open notepad",
            "delete file test.txt",
            "reboot system",
            "shutdown",
            "format c:",
            "list files",
            "show active window",
        ]
        for cmd in invalid_cmds:
            with self.subTest(cmd=cmd):
                self.assertFalse(self.skill.can_handle(cmd))

    def test_03_parse_command_structured_and_natural_language(self) -> None:
        """Verify parse_command normalizes dict and text inputs correctly."""
        # Dict parsing
        op, target, params, _ = self.skill.parse_command({
            "operation": "open_url",
            "parameters": {"url": "https://example.com"},
        })
        self.assertEqual(op, "open_url")
        self.assertEqual(params.get("url"), "https://example.com")

        # Natural language open_url
        op, target, params, _ = self.skill.parse_command("open url https://google.com")
        self.assertEqual(op, "open_url")
        self.assertEqual(target, "https://google.com")
        self.assertEqual(params.get("url"), "https://google.com")

        # Natural language domain command
        op, target, params, _ = self.skill.parse_command("open google.com")
        self.assertEqual(op, "open_url")
        self.assertEqual(target, "https://google.com")
        self.assertEqual(params.get("url"), "https://google.com")

        # Natural language search_web
        op, target, params, _ = self.skill.parse_command("search web for rust lang")
        self.assertEqual(op, "search_web")
        self.assertEqual(target, "rust lang")
        self.assertEqual(params.get("query"), "rust lang")

        # Natural language open_browser
        op, target, params, _ = self.skill.parse_command("open browser")
        self.assertEqual(op, "open_browser")

    # -----------------------------------------------------------------------
    # 2. URL Validation Tests (Strict Allowlist: http:// and https://)
    # -----------------------------------------------------------------------

    def test_04_valid_https_url(self) -> None:
        """Verify valid HTTPS URLs pass validation and launch browser."""
        urls = [
            "https://example.com",
            "https://www.google.com/search?q=jarvis",
            "https://sub.domain.org/path/to/resource?id=123#heading",
            "https://github.com/openai/whisper",
        ]
        for u in urls:
            with self.subTest(url=u):
                self.mock_open.reset_mock()
                res = self.skill.open_url(u)
                self.assertTrue(res["opened"])
                self.assertEqual(res["url"], u)
                self.mock_open.assert_called_once_with(u)

    def test_05_valid_http_url(self) -> None:
        """Verify valid HTTP URLs pass validation and launch browser."""
        urls = [
            "http://example.com",
            "http://localhost:8080/dashboard",
            "http://127.0.0.1:5000/api/v1/health",
        ]
        for u in urls:
            with self.subTest(url=u):
                self.mock_open.reset_mock()
                res = self.skill.open_url(u)
                self.assertTrue(res["opened"])
                self.assertEqual(res["url"], u)
                self.mock_open.assert_called_once_with(u)

    def test_06_empty_and_whitespace_urls_rejected(self) -> None:
        """Verify empty, None, and whitespace-only URLs are rejected safely."""
        invalid_urls = [
            "",
            "   ",
            "\t\n",
            None,
        ]
        for u in invalid_urls:
            with self.subTest(url=u):
                with self.assertRaises(SkillExecutionError):
                    validate_url(u)
                with self.assertRaises(SkillExecutionError):
                    self.skill.open_url(u)  # type: ignore

    def test_07_javascript_scheme_rejected(self) -> None:
        """Verify javascript: URLs are strictly rejected as security violations."""
        bad_urls = [
            "javascript:alert(1)",
            "javascript:void(0)",
            "JAVASCRIPT:alert(document.cookie)",
            "javascript://%0aalert(1)",
        ]
        for u in bad_urls:
            with self.subTest(url=u):
                with self.assertRaises(SecurityPolicyViolationError):
                    validate_url(u)
                with self.assertRaises(SecurityPolicyViolationError):
                    self.skill.open_url(u)
        self.mock_open.assert_not_called()

    def test_08_file_scheme_rejected(self) -> None:
        """Verify file: URLs are strictly rejected as security violations."""
        bad_urls = [
            "file:///C:/Windows/System32/calc.exe",
            "file:///etc/passwd",
            "FILE://localhost/c$/boot.ini",
        ]
        for u in bad_urls:
            with self.subTest(url=u):
                with self.assertRaises(SecurityPolicyViolationError):
                    validate_url(u)
                with self.assertRaises(SecurityPolicyViolationError):
                    self.skill.open_url(u)
        self.mock_open.assert_not_called()

    def test_09_data_scheme_rejected(self) -> None:
        """Verify data: URLs are strictly rejected as security violations."""
        bad_urls = [
            "data:text/html,<script>alert(1)</script>",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
            "DATA:application/javascript;base64,YWxlcnQoMSk=",
        ]
        for u in bad_urls:
            with self.subTest(url=u):
                with self.assertRaises(SecurityPolicyViolationError):
                    validate_url(u)
                with self.assertRaises(SecurityPolicyViolationError):
                    self.skill.open_url(u)
        self.mock_open.assert_not_called()

    def test_10_unsupported_and_custom_schemes_rejected(self) -> None:
        """Verify all other arbitrary/custom schemes are rejected."""
        bad_urls = [
            "vbscript:msgbox(1)",
            "about:blank",
            "chrome://settings",
            "edge://flags",
            "ms-settings:windowsupdate",
            "blob:https://example.com/uuid",
            "ftp://ftp.example.com/file.zip",
            "ws://localhost:9000/socket",
            "wss://secure.example.com",
            "ssh://user@remote.host",
            "telnet://telehack.com",
            "mailto:admin@example.com",
            "custom://payload",
        ]
        for u in bad_urls:
            with self.subTest(url=u):
                with self.assertRaises(SecurityPolicyViolationError):
                    validate_url(u)
                with self.assertRaises(SecurityPolicyViolationError):
                    self.skill.open_url(u)
        self.mock_open.assert_not_called()

    def test_11_malformed_urls_rejected(self) -> None:
        """Verify malformed URLs without valid scheme, host, or with unencoded whitespace fail."""
        malformed_urls = [
            "not-a-url",
            "example.com",
            "http://",
            "https://",
            "https:///only-path",
            "http://example .com",
            "https://example.com/path with space",
            "://missing-scheme.com",
        ]
        for u in malformed_urls:
            with self.subTest(url=u):
                with self.assertRaises(SkillExecutionError):
                    validate_url(u)
                with self.assertRaises(SkillExecutionError):
                    self.skill.open_url(u)
        self.mock_open.assert_not_called()

    def test_12_unc_paths_and_control_characters_rejected(self) -> None:
        """Verify UNC paths and null/control characters raise SecurityPolicyViolationError."""
        dangerous_urls = [
            r"\\server\share\exploit.html",
            r"//server/share/index.html",
            "https://example.com\x00/admin",
            "https://example.com\r\nHeader: Injection",
            "https://example.com\x1b[31m",
        ]
        for u in dangerous_urls:
            with self.subTest(url=u):
                with self.assertRaises(SecurityPolicyViolationError):
                    validate_url(u)
                with self.assertRaises(SecurityPolicyViolationError):
                    self.skill.open_url(u)
        self.mock_open.assert_not_called()

    # -----------------------------------------------------------------------
    # 3. Web Search Tests
    # -----------------------------------------------------------------------

    def test_13_search_web_normal_query(self) -> None:
        """Verify search_web executes with a normal query and validates URL."""
        res = self.skill.search_web("Python")
        self.assertTrue(res["opened"])
        self.assertEqual(res["query"], "Python")
        self.assertEqual(res["engine"], "google")
        expected_url = "https://www.google.com/search?q=Python"
        self.assertEqual(res["url"], expected_url)
        self.mock_open.assert_called_once_with(expected_url)

    def test_14_search_web_with_spaces(self) -> None:
        """Verify search_web safely encodes spaces in search query."""
        res = self.skill.search_web("antigravity autonomous agent")
        self.assertTrue(res["opened"])
        self.assertEqual(res["query"], "antigravity autonomous agent")
        expected_encoded = urllib.parse.urlencode({"q": "antigravity autonomous agent"})
        expected_url = f"https://www.google.com/search?{expected_encoded}"
        self.assertEqual(res["url"], expected_url)
        self.mock_open.assert_called_once_with(expected_url)

    def test_15_search_web_with_special_characters(self) -> None:
        """Verify search_web properly escapes special characters (&, ?, #, /, quotes, unicode)."""
        queries = [
            "C++ & C# programming",
            "what is 100% of 50? / 2",
            'search "exact phrase" #hashtag',
            "こんにちは 世界 <test>",
            "cmd.exe | powershell -enc",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.mock_open.reset_mock()
                res = self.skill.search_web(q)
                self.assertTrue(res["opened"])
                self.assertEqual(res["query"], q)
                # Verify that the generated URL parses cleanly and parameter decodes back
                parsed = urllib.parse.urlsplit(res["url"])
                self.assertEqual(parsed.scheme, "https")
                self.assertEqual(parsed.netloc, "www.google.com")
                params = urllib.parse.parse_qs(parsed.query)
                self.assertEqual(params.get("q"), [q])
                self.mock_open.assert_called_once_with(res["url"])

    def test_16_search_web_alternative_engines(self) -> None:
        """Verify search_web works across supported search engines (bing, duckduckgo)."""
        engines = [
            ("bing", "https://www.bing.com/search?q=AI"),
            ("duckduckgo", "https://duckduckgo.com/?q=AI"),
            ("ddg", "https://duckduckgo.com/?q=AI"),
        ]
        for eng, expected_prefix in engines:
            with self.subTest(engine=eng):
                self.mock_open.reset_mock()
                res = self.skill.search_web("AI", engine=eng)
                self.assertTrue(res["opened"])
                self.assertEqual(res["url"], expected_prefix)
                self.mock_open.assert_called_once_with(expected_prefix)

    def test_17_search_web_empty_query_rejected(self) -> None:
        """Verify empty or whitespace-only search query raises SkillExecutionError."""
        for empty_q in ["", "   ", "\t"]:
            with self.subTest(query=empty_q):
                with self.assertRaises(SkillExecutionError):
                    self.skill.search_web(empty_q)
        self.mock_open.assert_not_called()

    # -----------------------------------------------------------------------
    # 4. Open Browser Tests
    # -----------------------------------------------------------------------

    def test_18_open_browser_default_landing_page(self) -> None:
        """Verify open_browser with no arguments opens DEFAULT_BROWSER_URL."""
        res = self.skill.open_browser()
        self.assertTrue(res["opened"])
        self.assertTrue(res["default"])
        self.assertEqual(res["url"], DEFAULT_BROWSER_URL)
        self.mock_open.assert_called_once_with(DEFAULT_BROWSER_URL)

    def test_19_open_browser_with_custom_url(self) -> None:
        """Verify open_browser with custom valid URL opens requested destination."""
        target = "https://python.org"
        res = self.skill.open_browser(target)
        self.assertTrue(res["opened"])
        self.assertFalse(res["default"])
        self.assertEqual(res["url"], target)
        self.mock_open.assert_called_once_with(target)

    def test_20_open_browser_with_invalid_url_rejected(self) -> None:
        """Verify open_browser with invalid custom URL raises error and does not open."""
        with self.assertRaises(SecurityPolicyViolationError):
            self.skill.open_browser("javascript:void(0)")
        self.mock_open.assert_not_called()

    # -----------------------------------------------------------------------
    # 5. Failure & Exception Handling Tests
    # -----------------------------------------------------------------------

    def test_21_mocked_webbrowser_open_returns_false(self) -> None:
        """Verify that when webbrowser.open() returns False, SkillExecutionError is raised."""
        self.mock_open.return_value = False
        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.open_url("https://example.com")
        self.assertIn("Browser launch returned failure", str(ctx.exception))

    def test_22_mocked_webbrowser_open_raises_exception(self) -> None:
        """Verify that when webbrowser.open() raises an unexpected exception, it is cleanly handled."""
        self.mock_open.side_effect = OSError("Browser process creation failed")
        with self.assertRaises(SkillExecutionError) as ctx:
            self.skill.open_url("https://example.com")
        self.assertIn("Failed to launch browser", str(ctx.exception))

    # -----------------------------------------------------------------------
    # 6. Execute Protocol & Event Telemetry Tests
    # -----------------------------------------------------------------------

    def test_23_execute_protocol_integration(self) -> None:
        """Verify execute() returns SystemSkillResult and broadcasts lifecycle events."""
        res = self.skill.execute({
            "operation": "open_url",
            "parameters": {"url": "https://example.org"},
        })
        self.assertIsInstance(res, SystemSkillResult)
        self.assertTrue(res.success)
        self.assertEqual(res.operation, "open_url")
        self.assertEqual(res.data["url"], "https://example.org")
        self.assertTrue(res.duration_ms >= 0.0)

        # Check emitted lifecycle events
        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillStarted, event_types)
        self.assertIn(SystemSkillCompleted, event_types)

    def test_24_execute_failure_event_broadcast(self) -> None:
        """Verify failed execution emits SystemSkillFailed event."""
        self.mock_open.return_value = False
        with self.assertRaises(SkillExecutionError):
            self.skill.execute({
                "operation": "open_url",
                "parameters": {"url": "https://example.org"},
            })

        event_types = [type(e) for e in self.events]
        self.assertIn(SystemSkillStarted, event_types)
        self.assertIn(SystemSkillFailed, event_types)

    # -----------------------------------------------------------------------
    # 7. Invariant & Security Verification (Zero Subprocess / Shell Execution)
    # -----------------------------------------------------------------------

    def test_25_never_imports_or_calls_subprocess_or_shells(self) -> None:
        """Statically verify browser_skills code does not import or invoke subprocess or shell commands."""
        import app.skills.system.browser_skills as bs_mod
        with open(bs_mod.__file__, "r", encoding="utf-8") as f:
            code_lines = [
                line for line in f.readlines()
                if not line.strip().startswith(('"""', "'''", "*", "-", "1.", "2.", "3.", "4.", "5.", "6.", "7."))
            ]
        code_body = "".join(code_lines)

        prohibited_tokens = [
            "import subprocess",
            "from subprocess",
            "os.system",
            "os.popen",
            "os.spawn",
            "shell=True",
            "cmd.exe",
            "powershell.exe",
            "pwsh",
            "taskkill",
            "TerminateProcess",
            "SetSuspendState",
            "ExitWindowsEx",
            "InitiateSystemShutdown",
        ]
        for tok in prohibited_tokens:
            with self.subTest(token=tok):
                self.assertNotIn(tok, code_body)

    def test_26_ast_import_verification(self) -> None:
        """Verify via AST that browser_skills only imports allowed standard library and app modules."""
        import app.skills.system.browser_skills as bs_mod
        with open(bs_mod.__file__, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=bs_mod.__file__)

        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split(".")[0])

        # Prohibited modules
        self.assertNotIn("subprocess", imported_modules)
        self.assertNotIn("ctypes", imported_modules)
        self.assertNotIn("shutil", imported_modules)

        # Permitted modules
        expected_subset = {"webbrowser", "urllib", "logging", "threading", "re", "typing", "app", "__future__"}
        self.assertTrue(imported_modules.issubset(expected_subset))

    def test_27_thread_safety_concurrent_calls(self) -> None:
        """Verify thread safety of BrowserSkills across concurrent requests."""
        def run_search(idx: int) -> bool:
            res = self.skill.search_web(f"query_{idx}")
            return bool(res["opened"])

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_search, i) for i in range(30)]
            results = [f.result() for f in futures]

        self.assertTrue(all(results))
        self.assertEqual(len(results), 30)
        self.assertEqual(self.mock_open.call_count, 30)

    def test_28_natural_language_domain_launch(self) -> None:
        """Verify natural language domain command (e.g. 'open google.com') launches with https."""
        res = self.skill.execute("open google.com")
        self.assertTrue(res.success)
        self.assertEqual(res.data["url"], "https://google.com")
        self.assertTrue(res.data["opened"])
        self.mock_open.assert_called_with("https://google.com")

    @patch("subprocess.Popen")
    def test_29_routing_integration_skill_manager_and_command_router(self, mock_popen: MagicMock) -> None:
        """Verify SkillManager and CommandRouter deterministically route URLs to BrowserSkills and apps to AppSkills."""
        from app.router.router import CommandRouter
        from app.skills.manager import SkillManager
        from app.skills.system.app_skills import AppSkills

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        manager = SkillManager(container_instance=self.container, auto_register_in_container=False)
        app_skill = AppSkills(
            security_policy=self.security_policy,
            confirmation_manager=self.confirmation_manager,
            event_bus=self.bus,
        )
        manager.register(app_skill)
        manager.register(self.skill)

        # 1. Test SkillManager direct execution
        # 'open google.com' must route to BrowserSkills
        res_browser = manager.execute("open google.com")
        self.assertTrue(res_browser.success)
        self.assertEqual(res_browser.data["url"], "https://google.com")
        self.mock_open.assert_called_with("https://google.com")
        self.assertFalse(mock_popen.called)

        # 'open notepad' must route to AppSkills
        res_app = manager.execute("open notepad")
        self.assertTrue(res_app.success)
        self.assertEqual(res_app.data["app_name"], "notepad")
        self.assertTrue(res_app.data["launched"])
        self.assertTrue(mock_popen.called)

        # 2. Test CommandRouter execution
        from app.core.event_bus import EventBus
        router_bus = EventBus()
        router = CommandRouter(
            container_instance=self.container,
            skill_manager_instance=manager,
            event_bus_instance=router_bus,
            auto_register_in_container=False,
        )

        self.mock_open.reset_mock()
        mock_popen.reset_mock()

        # Route 'open google.com' via CommandRouter
        router_res_browser = router.route("open google.com")
        self.assertTrue(router_res_browser.success)
        self.assertEqual(router_res_browser.data["url"], "https://google.com")
        self.mock_open.assert_called_with("https://google.com")
        self.assertFalse(mock_popen.called)

        # Route 'open notepad' via CommandRouter
        router_res_app = router.route("open notepad")
        self.assertTrue(router_res_app.success)
        self.assertEqual(router_res_app.data["app_name"], "notepad")
        self.assertTrue(router_res_app.data["launched"])
        self.assertTrue(mock_popen.called)


if __name__ == "__main__":
    unittest.main()

