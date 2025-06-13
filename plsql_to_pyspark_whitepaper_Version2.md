# White Paper: Technical Deep Dive into Robust PL/SQL Parsing, Chunking, and Conversion Codebase

## Abstract

This document provides a thorough technical explanation of a Python-based system for parsing, chunking, and converting legacy PL/SQL code to PySpark, with a special focus on auditability, chunk stability, and user-driven workflow. The codebase leverages Streamlit for interactive UI, robust parsing functions for PL/SQL, and LLM-based code conversion, all engineered to meet enterprise-grade requirements for migration, traceability, and compliance.

---

## 1. Introduction

Legacy PL/SQL code presents significant challenges for modernization initiatives. The codebase under discussion implements a best-practice approach for transforming such code into PySpark, focusing on the following engineering challenges:

- **Parsing**: Safely dissecting PL/SQL scripts into logical blocks, respecting nested constructs and top-level delimiters.
- **Chunking**: Breaking down large blocks into manageable units for LLM processing.
- **Classification**: Labeling blocks for traceability and audit.
- **One-time, controlled processing**: Guaranteeing stability and reproducibility in user-facing applications.
- **Conversion and Validation**: Using LLMs to generate PySpark and statically validate outputs.

---

## 2. Detailed Explanation of Code and Logic

### 2.1. Imports and Dependencies

- **os, io, re**: Standard Python modules for file, stream, and regex operations.
- **streamlit as st**: For building a rich, interactive web UI.
- **pandas as pd**: For tabular data manipulation and CSV export.
- **sqlparse**: For parsing and tokenizing SQL/PLSQL.
- **typing**: For type hinting, improving readability and static analysis.
- **Path**: For robust file path handling.

---

### 2.2. PL/SQL Parsing Agent

#### 2.2.1. Regular Expressions

- `_TERMINATOR`: Matches a line with only a `/` (standard block delimiter in Oracle).
- `_BEGIN` and `_END`: Match lines containing `BEGIN` and `END` (case-insensitive), essential for tracking block nesting.
- `_HEADER_RE`: Captures block-defining statements like `CREATE OR REPLACE PROCEDURE`, `FUNCTION`, etc., for classification.

#### 2.2.2. Comment Removal

```python
def remove_comments(src: str) -> str:
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    src = re.sub(r'--.*', '', src)
    return src
```
- **Purpose**: Prevents false positives in block detection due to comments containing `/`, `BEGIN`, or `END`.
- **Robustness**: Handles both multi-line and single-line comments using regex.

#### 2.2.3. Top-Level Block Splitting

```python
def split_top_level(src: str) -> List[str]:
    blocks, buf, depth = [], [], 0
    for ln in src.splitlines():
        if _BEGIN.search(ln): depth += 1
        if _END.search(ln):   depth = max(depth-1, 0)
        buf.append(ln)
        if _TERMINATOR.match(ln) and depth == 0:
            blocks.append('\n'.join(buf).rstrip())
            buf.clear()
    if buf:
        blocks.append('\n'.join(buf).rstrip())
    return blocks
```
- **Innovation**: Tracks `depth` to correctly interpret `/` only at the outermost level.
- **Result**: Only splits when not inside nested blocks—preserving code integrity.

#### 2.2.4. Safe Sub-Chunking

```python
def safe_split(block: str, max_lines:int) -> List[str]:
    if len(lines) <= max_lines:
        return [block]
    # ... [splitting logic] ...
```
- **Purpose**: Avoids overwhelming LLM input limits by chunking large blocks, but only at logical boundaries (depth==0, after `;` or `/`).
- **Benefit**: Each sub-chunk is independently valid, never splitting statements or inner blocks.

#### 2.2.5. Block Classification

```python
def classify(block:str) -> str:
    m = _HEADER_RE.search(block)
    if m:
        return m.group(2).upper().replace(' ', '_')
    if block.lstrip().upper().startswith('BEGIN'):
        return 'ANONYMOUS_BLOCK'
    return 'UNKNOWN'
```
- **Purpose**: Facilitates mapping, downstream conversion, and audit by structurally tagging every chunk.

#### 2.2.6. Orchestration: process_plsql_string

```python
def process_plsql_string(source:str, max_lines:int=400) -> List[Dict]:
    clean  = remove_comments(source)
    lvl1   = split_top_level(clean)
    chunks: List[Dict] = []
    for blk in lvl1:
        for sub in safe_split(blk, max_lines):
            chunks.append(sub)
    return [
        {"id": f"blk_{i+1:03}", "type": classify(c), "code": c}
        for i, c in enumerate(chunks)
    ]
```
- **Pipeline**: Remove comments → split top-level → safe sub-chunk large blocks → classify.
- **Output**: Each chunk is a dict with stable ID, type, and code.

#### 2.2.7. File Wrapper

```python
def process_plsql_file(path:str, max_lines:int=400) -> List[Dict]:
    return process_plsql_string(Path(path).read_text(encoding='utf-8'), max_lines)
```
- **Flexibility**: Accepts both string and file inputs.

---

### 2.3. PySpark Code Validation

```python
def validate_pyspark_code(code: str) -> Tuple[bool, List[str]]:
    # ... [static checks] ...
```
- **Static Checks**: Looks for SparkSession usage, DataFrame API, bad anti-patterns, syntax, and imports.
- **Enterprise Value**: Ensures LLM output meets minimum PySpark standards before further use.

---

### 2.4. LLM Provider Framework

- **Abstract Class**: `LLMProvider` for future extensibility.
- **GeminiProvider & OpenAIProvider**: 
  - Unified interface for block-wise and full-script conversion.
  - Uses user-provided credentials and deployment details.
  - Handles exceptions gracefully, returning error annotations.

---

### 2.5. UI and State Management

#### 2.5.1. Streamlit UI Helpers

- **User Profile**: Visual branding, can be replaced by enterprise identity.
- **UI Styling**: Custom CSS for modern, branded appearance.

#### 2.5.2. Chunking and Conversion Logic

```python
def get_structured_chunks(sql_code, max_lines=400):
    if "last_sql_code" not in st.session_state or st.session_state["last_sql_code"] != sql_code:
        blocks = process_plsql_string(sql_code, max_lines=max_lines)
        st.session_state["last_sql_code"] = sql_code
        st.session_state["chunked_blocks"] = blocks
    return st.session_state.get("chunked_blocks", [])
```
- **Key Design**: Ensures chunking is performed only once per user input, cached in session. Prevents accidental re-chunking or instability.

#### 2.5.3. User-Driven Workflow

- **Conversion is only triggered by explicit user action (button click)**.
- **Chunked data and converted outputs are cached separately**; conversion is never recursive or automatic.
- **Audit-friendly DataFrame and CSV export**: Includes Sr No, Chunk ID, Block Type, line counts, and code.

---

### 2.6. End-to-End Data Flow

1. **User uploads or pastes PL/SQL code**.
2. **Code is parsed and chunked once** (unless changed), each chunk is labeled and tracked.
3. **User can view chunk breakdown and metadata**.
4. **User initiates conversion**, which processes each chunk via LLM and validator.
5. **Preview and audit of mapping**, with CSV export.
6. **Optional: User can also generate optimized, unified PySpark code for the whole script**.

---

## 3. Engineering Rationale

- **Correctness**: Avoids splitting code in invalid places, preserving business logic.
- **Auditability**: Every chunk/block is uniquely identified, classified, and traceable from input through conversion and validation.
- **Efficiency**: No redundant computation—parsing and conversion are only redone when necessary.
- **User Experience**: Predictable, responsive, with clear separation of steps and audit trail.
- **Extensibility**: Modular design allows new block types, LLMs, and validation rules to be added with ease.

---

## 4. Compliance and Enterprise Readiness

- **CSV download includes all metadata for downstream audit and compliance.**
- **Session state prevents accidental double-processing, supporting reproducible workflows.**
- **Static validation ensures quality of generated PySpark before use in production or migration.**

---

## 5. Conclusion

This codebase embodies best practices for the safe, auditable, and efficient migration of PL/SQL workloads to modern big data frameworks, with a focus on transparency, user control, and enterprise requirements. The design choices—especially one-time chunking, explicit user action for conversion, and stable metadata—distinguish it from naive or batch-oriented approaches, making it especially suitable for regulated, large-scale environments.

---

**Appendix: Key Design Patterns**
- **Depth-aware parsing:** Tracks block nesting to prevent accidental logic splits.
- **Session state caching:** Ensures stability and performance.
- **Abstract provider pattern:** LLM abstraction for extensibility.
- **Fail-safe validation:** Catches conversion errors early.
