# Detailed Logic & Explanation of the PL/SQL to PySpark Converter

This Streamlit application is a rich UI tool designed to convert PL/SQL scripts to PySpark DataFrame API code using LLMs (Gemini or Azure OpenAI). Below is a detailed step-by-step explanation of each core logic block, design choices, and why the parsing and chunking agent works as it does.

---

## 1. **PL/SQL Chunking & Parsing Agent: Why This Design?**

### **Challenges in PL/SQL to PySpark Conversion**
- PL/SQL scripts can be large, with complex procedures, functions, and standalone statements.
- LLMs have input size restrictions (context window), so sending a whole file at once can fail or produce suboptimal results.
- Quality improves when breaking code into logical, manageable blocks.
- Blocks must be meaningful: splitting mid-procedure or mid-statement is bad. But also, not all files use clear delimiters, so multiple strategies are required.

### **Multi-Strategy Chunking Logic**

#### **a. Delimiter-based Splitting**
- **Why:** PL/SQL scripts often use `/` on a line by itself to separate CREATE statements, procedures, or functions.
- **Implementation:** `_split_by_delimiter` splits at these lines, preserving logical blocks.
- **Fallback:** If only one block is found, it means delimiters weren't used, so try the next approach.

#### **b. Regex-based Logical Block Splitting**
- **Why:** Not all PL/SQL scripts use `/`, but most use keywords like `CREATE [OR REPLACE]`, `FUNCTION`, `PROCEDURE`, `PACKAGE`, `TRIGGER`, `DECLARE`, or `BEGIN`.
- **Implementation:** `_regex_chunk_blocks` uses regex to find these blocks, splitting at `END;` boundaries or semicolons for simple statements. This produces logical code chunks.
- **Sub-chunking:** If a block is too large (over `max_chunk_size`), `_ast_chunk_blocks` tries to split further, e.g., at `BEGIN`.
- **Cleaning:** Small blocks are grouped for efficiency (small statements are batched if total size < threshold).

#### **c. Fallback: Fixed Line Chunking**
- **Why:** If no logical boundaries are found, chunk the code by a fixed number of lines (default 100).
- **Implementation:** `_chunk_by_lines` is a last resort for very unstructured code.

#### **d. Main Entry Point**
- **Function:** `split_plsql_into_blocks`
- **Logic:** Tries delimiter, then regex, then fallback, returning the best possible block division.

#### **Result:**  
- Chunks are logical, manageable, and LLM-friendly, maximizing the chance of high-quality conversion and minimizing hallucination or missed logic.

---

## 2. **LLM Provider Abstraction**

### **Why an Abstract Provider Pattern?**
- Supports future addition of new LLMs with minimal code changes.
- Cleanly separates UI, logic, and API-specific quirks.

### **Providers Implemented**
- **GeminiProvider:** For Google Gemini API, uses `generate_content`.
- **OpenAIProvider:** For Azure OpenAI, uses `ChatCompletion.create`.
- Both have:
  - `convert`: Per-block conversion prompt (chunk-wise).
  - `convert_optimized`: Full-script, single optimized code conversion.

---

## 3. **Validation Agent for PySpark Code**

### **Why Validate?**
- LLMs may hallucinate, skip imports, or output non-executable code.
- Users benefit from instant feedback on likely issues.

### **How?**
- **Function:** `validate_pyspark_code`
- **Checks for:**
  - SparkSession usage (did user forget to initialize Spark?).
  - DataFrame API patterns (ensures code isn't just plain Python).
  - RDD/SQLContext (flags for anti-patterns).
  - Unmatched parentheses (quick syntax sanity).
  - PySpark import statements.
- **Result:** Returns validation status and comments, which are shown above the generated code.

---

## 4. **LLM Credential UI & Validation**

### **Purpose:**
- Securely collect and validate user API keys for Gemini or Azure OpenAI.
- Prevents invalid keys from causing silent failures later.

### **Flow:**
1. User selects LLM provider and enters credentials.
2. On submit, a minimal API call is made to verify the credentials.
3. Result shown; if valid, credentials are cached in session.

---

## 5. **User Interface Flow**

- **Sidebar:** User profile, LLM credentials, input method (file upload or paste).
- **Main Area:**
  1. **Original Code:** Displays the PL/SQL code for reference.
  2. **Chunking:** Code is chunked and stored in session for efficiency.
  3. **Conversion:** Each block is sent to the LLM for conversion, and validated.
  4. **Preview Table:** Shows a side-by-side view of PL/SQL blocks and corresponding PySpark code (with validation comments).
  5. **CSV Download:** Allows user to download the mapping as a CSV.
  6. **Optimized Conversion:** Option to produce a single, holistic, production-ready PySpark script for the entire input (not just concatenated blocks).
  7. **Final Output:** Shows the optimized code with validation, and allows download as `.py`.

---

## 6. **Styling & Polish**

- **Custom CSS:** Rich, modern look using Streamlit's `unsafe_allow_html`.
- **Profile Card:** Adds a professional personal touch.
- **Buttons, Tables, and Code Blocks:** Styled for visual clarity and emphasis.

---

## 7. **Session State Handling**

- **Why:** Prevents redundant API calls, preserves state between reruns, and speeds up UX.
- **How:** Stores raw SQL, parsed blocks, converted blocks, validation results, and final output in `st.session_state`.

---

## 8. **Summary of Why the Parsing/Chunking Agent Works Like This**

- **Goal:** Provide the best possible input to LLMs for code translation, respecting both LLM context windows and logical code boundaries.
- **Approach:** Layered chunking (delimiter → regex → fallback) ensures almost any script can be robustly and meaningfully split.
- **Result:** Maximizes translation quality, reduces hallucinations, and produces code that's easier to debug and validate post-conversion.

---

## 9. **Extensibility**

- New LLM providers can be added by subclassing `LLMProvider`.
- Chunking logic can be tuned for specific PL/SQL dialects or other SQL-like languages.
- Additional validation (for more PySpark/SQL patterns) can be layered in without UI changes.

---

**In essence, the code is designed to robustly parse, chunk, convert, and validate PL/SQL scripts for migration to PySpark, with a strong focus on usability, reliability, and developer experience.**
