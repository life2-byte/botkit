---
name: writing-documents
description: Use when asked to write a report, memo, proposal, letter, plan, article or any document the user will keep or send.
---
1. Work out the audience and purpose first. Then choose a structure: title, short summary at the top, headings, then the detail.
2. Lead with the conclusion or the ask. Keep paragraphs short, use bullets only for parallel items, use a table for comparisons.
3. Be specific: real names, numbers and dates from the user or from sources. Never pad with generic filler.
4. If the user wants a file, write the content in simple markdown and create it with `create_docx` (headings with #, bullets with -, **bold**, | tables |).
5. Read the file back with `read_file` and fix mistakes. Tell the user the file name.
6. Match the user's language and tone. For emails and messages, just give the text; no file needed.
