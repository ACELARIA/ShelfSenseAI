We are refactoring ShelfSense AI.

Current pipeline

Upload

↓

OCR

↓

Raw OCR Text

↓

Qwen

↓

Recommendation

Target pipeline

Upload

↓

Preprocessing

↓

OCR

↓

Field Extraction

↓

Structured Product JSON

↓

Qwen

↓

Recommendation

Guidelines

Do NOT rewrite the entire project.

Preserve existing APIs where possible.

Keep services independent.

Generate code incrementally.

Never generate multiple files unless explicitly requested.