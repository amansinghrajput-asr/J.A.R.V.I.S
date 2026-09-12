# Phase 22.3: File & Folder Skills

> **Status**: COMPLETE & VERIFIED  
> **Release**: `v0.22.3`  
> **Prerequisites**: Phase 22.1 (Foundation & Security) & Phase 22.2 (Application Skills) Verified  
> **Scope**: Sprint 22.3 File & Folder Skills ONLY

---

## 1. Executive Summary

Phase 22.3 implements the **File & Folder Skills** (`FileSkills`) subsystem, granting J.A.R.V.I.S. safe, controlled, and bounded local filesystem automation across Windows and desktop environments.

All 9 operations inherit from `BaseSystemSkill`, enforce canonical path resolution via `validate_path()`, guard against path traversal attacks, protect critical Windows system folders, mandate interactive confirmation for all destructive changes (deletion, moving, overwriting), and strictly prevent directory deletion recursion traps and symlink escapes.

---

## 2. Supported Operations

| Operation | Safety Tier | Confirmation Required | Description |
|---|---|---|---|
| `create_folder` | `SAFE` | No | Creates a directory tree (`mkdir -p`). Returns gracefully if already existing. |
| `create_file` | `SAFE` / `CONFIRMATION_REQUIRED` | Only if `overwrite=True` | Creates a new UTF-8 text file. Rejects silent overwriting of existing files unless confirmed. |
| `read_file` | `SAFE` | No | Reads file content with UTF-8 decoding (Latin-1 fallback) bounded by maximum read limit. |
| `list_directory`| `SAFE` | No | Non-recursive directory enumeration returning structured metadata (name, is_dir, size_bytes, is_symlink). |
| `rename_path` | `SAFE` / `CONFIRMATION_REQUIRED` | Only if destination exists | Renames a file or directory. Rejects silent overwrites of existing destination unless confirmed. |
| `move_path` | `CONFIRMATION_REQUIRED` | **Yes** | Moves a file or directory across paths or drive roots with mandatory confirmation. |
| `copy_path` | `SAFE` / `CONFIRMATION_REQUIRED` | Only if destination exists | Copies files or directories safely without following dangerous symlinks. |
| `delete_path` | `CONFIRMATION_REQUIRED` | **Yes** | Deletes a file, unlinks a symlink, or removes a directory with mandatory single-use confirmation token. |
| `open_in_explorer` | `SAFE` | No | Reveals the specified file or directory in Windows File Explorer without using a shell. |

---

## 3. Path Security & Traversal Protection

All paths (including source and destination parameters) are passed through `validate_path()` before any filesystem touch:

1. **Canonical Path Resolution**:
   Resolves path components (`Path.expanduser().resolve()`) to evaluate the true physical target regardless of relative segments or link indirection.
2. **Path Traversal Protection**:
   Rejects `..` path escapes when `allowed_roots` is configured.
3. **Prohibited Reserved Device Names**:
   Blocks legacy Windows device namespaces (`CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`).
4. **Protected Windows Locations**:
   Hard-blocks operations targeting Windows system directories (`C:\Windows`, `System32`, `SysWOW64`, `WinSxS`, `ProgramData\Microsoft`, `WindowsApps`, `$Recycle.Bin`).
5. **Filesystem Root Deletion Protection**:
   Drive roots (`C:\`, `D:\`, `/`) are hard-coded as `RESTRICTED` for deletion or destructive move operations.
6. **No Shell Invocations**:
   Zero shell utilities (`del`, `rmdir`, `rm`, `Remove-Item`) are used. All actions rely exclusively on Python standard library `pathlib`, `os`, and `shutil` APIs.

---

## 4. Symlink, Junction & Reparse Point Safety

Windows directory junctions and symbolic links can lead to catastrophic unintended deletions if tools blindly traverse them.

`FileSkills` applies strict link safety:
- **During Deletion (`delete_path`)**:
  Inspects `p.is_symlink()` and `os.path.islink(p)`. If the target is a link or junction, `os.unlink()` or `os.rmdir()` is invoked to destroy the link itself **without following into or deleting content within the linked target folder**.
- **During Copying (`copy_path`)**:
  `shutil.copy2(..., follow_symlinks=False)` and `shutil.copytree(..., symlinks=False)` ensure that links are not blindly traversed during duplicate creation.

---

## 5. Bounded Operational Limits

To prevent resource exhaustion and unbounded operations:
- **Maximum File Read Size (`max_read_bytes`)**: Default 5 MB (`5,242,880` bytes). Files exceeding this limit raise `SkillExecutionError` instead of causing out-of-memory errors.
- **Maximum Directory Entries (`max_dir_entries`)**: Default 100 entries, hard cap at 1,000 entries. Returns `has_more=True` if directory exceeds limit.
- **Maximum Recursive Depth (`max_recursive_depth`)**: Default 5 sub-levels for recursive deletion.
- **Maximum Recursive Items (`max_recursive_items`)**: Default 500 items for recursive deletion.

---

## 6. Confirmation & Overwrite Model

Destructive filesystem operations require an unconsumed, cryptographically parameter-bound token from `SystemConfirmationManager`:

```python
# 1. Requesting deletion triggers confirmation requirement:
res = skill.execute({"operation": "delete_path", "target": "data.csv"})
# Raises ConfirmationRequiredError with token 'sys_conf_8f3a9e...'

# 2. Operator confirms:
confirmation_mgr.resolve_confirmation(token, approved=True)

# 3. Execution proceeds with token:
res = skill.execute({
    "operation": "delete_path",
    "target": "data.csv",
    "confirmation_id": token,
})
# Deletion succeeds, token is consumed and purged
```

---

## 7. Performance Benchmark Results

Measured on Windows 11 with `benchmarks/file_skills_perf.py`:

| Benchmark Item | Measured Latency | Architectural Budget | Status |
|---|---|---|---|
| **Path Validation** | **448.41 µs** | < 1000.0 µs (1.0 ms) | **PASS** |
| **File Skill Lookup & `can_handle`** | **2.65 µs** | < 50.0 µs | **PASS** |
| **Security Policy Evaluation** | **500.32 µs** | < 1000.0 µs (1.0 ms) | **PASS** |
| **`create_folder` Dispatch** | **1,057.34 µs** | < 5000.0 µs (5.0 ms) | **PASS** |
| **`create_file` Dispatch** | **3,119.21 µs** | < 10000.0 µs (10.0 ms)| **PASS** |
| **`read_file` Dispatch** | **1,364.96 µs** | < 5000.0 µs (5.0 ms) | **PASS** |
| **`list_directory` Dispatch** | **21,322.80 µs** | < 50000.0 µs (50.0 ms)| **PASS** |

---

## 8. Verification & Regression Testing

### 8.1 Focused File Skills Test Suite (`tests/test_file_skills.py`)
- **30/30 PASS** in **0.276s**
- Validated: creation, reading, listing, renaming, moving, copying, deleting, symlink unlinking, overwrite confirmation, path traversal rejection, Windows device name rejection, root deletion rejection, explorer opening, event lifecycle, and concurrency.

### 8.2 System Test Discovery (`test_system*.py`)
- **42/42 PASS** in **0.181s**

### 8.3 Metacognition Regression Suite (`test_metacognition_*.py`)
- **88/88 PASS** in **3.333s**

### 8.4 Repository-Wide Full Regression Suite
- **889/889 PASS** in **20.1s** (821 baseline + 20 Sprint 22.1 + 18 Sprint 22.2 + 30 Sprint 22.3).
- Zero regressions across the entire repository.
