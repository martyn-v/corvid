"""Text normalization shared by the renderer's validator and the scorer."""

import unicodedata


def fold(text: str) -> str:
    """Casefolded, diacritic-stripped text, so 'Bogota' matches 'Bogotá'.

    Models transliterate; that is not an error.
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    )


def city(name: str) -> str:
    """The comparable city part; ground truth names read "Bogotá, Colombia"."""
    return fold(name.split(",")[0].strip())
