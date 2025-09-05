import re


BASIC_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'-\"\n()[]")

def clean_text(s: str) -> str:
    s = s.lower()
    cleaned = []
    for ch in s:
        if ch in BASIC_CHARS:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("\n")
        else:
            cleaned.append(" ")

    text = re.sub(r"\n{3,}","\n\n","".join(cleaned))
    text = re.sub(r"[ ]{2,}"," ", text)
    return text.strip()

raw = "Hello!!\tWorld###   How ARE   you?? 😀🔥"

print("Raw text:")
print(raw)

print("\nCleaned text:")
print(clean_text(raw))





