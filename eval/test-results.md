# Test Results

- Date: 2026-07-31
- Test runner: Python `unittest`
- Scope: `back-end/tests/test_*.py`
- Result: **PASS**
- Tests run: 85
- Passed: 85
- Failed: 0
- Errors: 0
- Duration: 15.109 seconds

## Command

Run from the `back-end/` directory:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## Notes

The run emitted one non-failing dependency warning:

```text
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is
deprecated; install `httpx2` instead.
```

The 50-query live golden evaluation was not included in this unit-test run.
That evaluation calls external APIs and may consume quota.
