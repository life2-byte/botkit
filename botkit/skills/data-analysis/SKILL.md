---
name: data-analysis
description: Use when asked to analyze, summarize, clean or compare data from a CSV, spreadsheet or list of numbers.
---
1. Look at the data first: columns, row count, types, missing values, obvious outliers. Use `read_file` (it reads .xlsx and .csv) or a script if a shell is available.
2. Compute results with code or `calculate`, never in your head. For large files use a script, not the conversation.
3. Sanity-check: do totals add up, are the units right, does the result make sense in size?
4. Report the answer first, then how you got it, then caveats (missing data, assumptions, small samples). Do not claim causes the data cannot show.
5. When a table or chart helps, save it as a file (`create_xlsx`, or a script's output) and name it.
