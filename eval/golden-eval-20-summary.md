# Golden Evaluation — 20 Questions

- Date: 2026-07-31
- Scope: Golden cases 1–20
- Pipeline: Full end-to-end dataset search
- Questions evaluated: 20
- Responses produced: 20
- Passed: 18
- Failed: 2
- Total verified dataset results returned: 142
- Overall pass rate: 90%

## Failed cases

### Case 14

**Question:** `image classification dataset for text images`

**Reason:** No verified dataset title or ID contained an expected known-good
pattern such as `text`, `document`, `ocr`, `word`, or `handwriting`.

### Case 16

**Question:** `text image classification training data`

**Reasons:**

- Parsed subject was `general`, but a text/document/OCR-related subject was expected.
- No verified dataset title or ID contained an expected known-good pattern.

## Result files

The complete machine-readable results are in `golden-eval-20.json`. Each of its
20 records includes:

- Original question
- Pass/fail result and failure reasons
- Pipeline response status
- Parsed intent
- Parse and ranking modes
- Pipeline warnings
- Candidate pool count
- Verified dataset answers with ID, title, score, and constraint status

## Reproduction command

Run from `back-end/`:

```powershell
..\.venv\Scripts\python.exe tests\run_eval.py --start 1 --count 20 --checkpoint ..\eval\golden-eval-20.json
```
