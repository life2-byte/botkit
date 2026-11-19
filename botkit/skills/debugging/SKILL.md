---
name: debugging
description: Use when something fails, throws an error, or behaves differently than expected.
---
1. Read the whole error message and traceback. The last lines and the first line of your own code in the trace matter most.
2. Reproduce it. If you cannot run the code, ask for the exact input and the full error text.
3. Form one hypothesis, find the evidence in the code (`search_files`, `read_file`), and confirm it before changing anything.
4. Make the smallest change that fixes the root cause. Change one thing at a time, then run again.
5. Explain in two or three sentences what was wrong and why the fix works.
6. If it still fails, report what you tried and what you learned instead of guessing again.
