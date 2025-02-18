import re
import json

# Path to your log file
logfile = "translation.log"

# This dictionary will map original text to its translation.
translations = {}

# Regular expression to capture blocks starting with a timestamp and "INFO - Translated:"
# The (?s) flag makes the dot match newlines; (?m) lets us use ^ to match line starts.
pattern = re.compile(
    r'(?m)^(?P<timestamp>\d{4}-\d{2}-\d{2} [\d:,]+) - INFO - Translated:\s*(?P<message>.*?)(?=^\d{4}-\d{2}-\d{2}|\Z)',
    re.DOTALL
)

with open(logfile, "r", encoding="utf-8") as f:
    log_content = f.read()

for match in pattern.finditer(log_content):
    message = match.group("message").strip()
    # We assume that your log format uses the arrow "→" to separate the original and translation.
    if "→" in message:
        original, translated = message.split("→", 1)
        original = original.strip()
        translated = translated.strip()
        # If the original already exists, you might want to verify or overwrite.
        translations[original] = translated

# Save the resulting dictionary to a JSON file.
with open("translations_cache.json", "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(translations)} translations to translations_cache.json")