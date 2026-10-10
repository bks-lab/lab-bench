"""Fixed German text normaliser for C2 (plan/c2.md), applied to reference
and hypothesis alike: Unicode NFKC, str.lower() (not casefold, so that
"ß" stays), every character of Unicode category P replaced by a space,
whitespace collapsed. Numbers are not spelled out."""
import re
import unicodedata

_WS = re.compile(r"\s+")


def norm_de(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = "".join(" " if unicodedata.category(c).startswith("P") else c for c in t)
    return _WS.sub(" ", t).strip()


if __name__ == "__main__":
    assert norm_de("Straße, „Hallo“ - Welt!") == "straße hallo welt", norm_de("Straße, „Hallo“ - Welt!")
    assert norm_de("  A B ") == "a b"
    print("ok")
