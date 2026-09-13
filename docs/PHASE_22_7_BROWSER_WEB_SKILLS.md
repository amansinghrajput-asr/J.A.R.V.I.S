# Phase 22.7: Browser & Web Skills Architecture & Specification

## 1. Overview & Architectural Integration

Phase 22.7 implements **Browser & Web Navigation Skills** (`BrowserSkills`) in J.A.R.V.I.S., allowing the assistant to safely navigate to web URLs, perform search queries across default search engines, and handle naked domains without invoking shell interpreters or exposing the host to local file execution vulnerabilities.

```
                  ┌───────────────────────────────┐
                  │    Planner / Command Router   │
                  └───────────────┬───────────────┘
                                  │ can_handle(query) / execute()
                                  ▼
                  ┌───────────────────────────────┐
                  │      BrowserSkills (p=60)     │
                  │   (subclasses BaseSystemSkill)│
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ URL Validation   │    │ Domain Routing & │    │ Standard Library    │
│ Strict Scheme    │    │ Normalization    │    │ webbrowser.open()   │
│ (http / https)   │    │ (auto-https)     │    │ (zero shell=True)   │
└──────────────────┘    └──────────────────┘    └─────────────────────┘
```

### Key Architectural Invariants
- **Subclasses `BaseSystemSkill`**: Seamlessly integrates into the Phase 22 skill hierarchy and dependency injection model.
- **Zero Shell Invocation**: Relies entirely on Python's standard library `webbrowser.open(url, new=2, autoraise=True)` rather than spawning `cmd.exe` or `powershell.exe`.
- **Strict Scheme Validation**: Strictly limits target navigation to `http://` and `https://`, blocking all URI-based local attack vectors.
- **Routing Boundary with AppSkills**: Cooperates with `AppSkills` using `is_url_or_domain` to ensure web targets (e.g. `"open google.com"`) route to `BrowserSkills` rather than failing in application allowlist validation.

---

## 2. Security Restrictions & URL Validation

### 2.1 Scheme Allowlist & Attack Prevention
`BrowserSkills` validates all URLs via `validate_url(url)` before dispatch:
- **Permitted Schemes**: `http://` and `https://` only.
- **Blocked Schemes**:
  - `file://` (prevents arbitrary local filesystem browsing or UNC SMB credential leaks).
  - `javascript:` (prevents script injection into active browser context).
  - `data:` (prevents inline payload and phishing execution).
  - `vbscript:`, `powershell:`, `shell:` (prevents command execution).
- **Malformed URIs**: Reject non-standard characters, incomplete netlocs, and empty hostnames.

### 2.2 Domain Recognition & Normalization
The `normalize_url(url_or_domain)` helper parses naked host inputs:
- Inputs like `"google.com"` or `"www.wikipedia.org"` are recognized via `is_url_or_domain()`.
- Valid domain targets are automatically prefixed with `https://` prior to browser dispatch.
- Inputs with executable file extensions (`.exe`, `.bat`, `.cmd`, `.msi`) are rejected from browser routing to preserve application launch integrity.

---

## 3. Supported Operations & Result Schemas

### 3.1 `open_url` / `browse_url` / `open_link`
Navigates the default web browser to a validated URL or normalized domain.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "action": "open_url",
  "url": "https://github.com",
  "success": true
}
```

### 3.2 `search_web` / `web_search`
Encodes a text query and opens the default search engine (Google by default).
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "action": "search_web",
  "query": "quantum computing algorithms",
  "url": "https://www.google.com/search?q=quantum+computing+algorithms",
  "success": true
}
```

### 3.3 `open_browser`
Launches the system default browser to its home page or default landing URL.
- **Classification**: `SAFE`
- **Output Schema**:
```json
{
  "action": "open_browser",
  "url": "https://www.google.com",
  "success": true
}
```

---

## 4. Planner Compatibility & Natural Language Routing

`BrowserSkills` is registered with priority 60 and integrates with the planning pipeline:
- **Planner Action Resolution**: Maps planner tasks with actions `open_url`, `browse_url`, `open_link`, `search_web`, `web_search`, `open_browser` directly to `BrowserSkills`.
- **Natural Language Parsing**: Recognizes voice and conversational prompts:
  - `"open youtube.com"`
  - `"search the web for latest space news"`
  - `"browse https://docs.python.org"`
  - `"open browser"`

---

## 5. Testing & Performance Validation

The implementation is verified via:
1. **Unit & Safety Tests (`tests/test_system_skills_browser.py`)**:
   - Validation of allowed and blocked URL schemes.
   - Naked domain normalization and scheme prefixing.
   - Natural language command parsing and action extraction.
   - Full mock isolation of `webbrowser.open` to eliminate internet traffic during testing.
2. **Performance Benchmarks (`benchmarks/browser_skills_perf.py`)**:
   - Evaluates URL validation throughput (~5 µs per check).
   - Domain routing evaluation (~1.8 µs per check).
   - Mocked browser dispatch latency (~55 µs).
   - 100% deterministic with zero external network connectivity.
