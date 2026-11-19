---
name: spreadsheets
description: Use when asked to build a spreadsheet, budget, tracker, schedule, or any table with calculations.
---
1. Design the columns first: one header row, one record per row, one fact per cell. No merged cells or blank spacer rows.
2. Use formulas for anything calculated (totals, averages, percentages), written as strings starting with "=". Do not paste computed numbers.
3. Keep inputs (rates, assumptions) in their own sheet or block and reference them from formulas.
4. Create the file with `create_xlsx` (sheets is a mapping of sheet name to rows).
5. Read it back with `read_file` and check that the formulas point at the right cells.
6. Tell the user the file name and what each sheet contains.
