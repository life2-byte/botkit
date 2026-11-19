---
name: presentations
description: Use when asked to make slides, a deck, or a presentation.
---
1. Ask nothing unless the topic is missing. Default to 6-10 slides.
2. Structure: title slide, agenda or the key message, one idea per content slide, a summary or next steps slide.
3. Each slide: a clear title that states the point, 3-5 bullets, each under about 12 words. No paragraphs on slides.
4. Put the longer explanation in the speaker notes.
5. Build it with `create_pptx` (slides is a list of objects with title, bullets, subtitle, notes).
6. Read it back with `read_file` to check slide count and text, then tell the user the file name.
