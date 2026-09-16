"""Shared text processing: tokenization used consistently by BM25 and embeddings.

One tokenizer for both datasets keeps the lexical (Q2) and semantic (Q3) retrieval
comparable -- any difference in results is due to the retrieval method, not to
divergent preprocessing.
"""
import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letters only, unicode-aware (keeps æøå, etc.)

# Small, hand-picked closed-class stopword lists. Deliberately short: BM25's IDF term
# already down-weights frequent words, so this only needs to strip the most extreme
# noise (articles, pronouns, auxiliaries), not do full linguistic stopword removal.
_EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "to", "from", "in",
    "on", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "can", "could", "this", "that",
    "these", "those", "it", "its", "as", "so", "than", "too", "not", "no", "he",
    "she", "they", "we", "you", "his", "her", "their", "our", "your", "i", "him",
}
_DA_STOPWORDS = {
    "og", "i", "jeg", "det", "at", "en", "den", "til", "er", "som", "på", "de",
    "med", "han", "af", "for", "ikke", "der", "var", "mig", "sig", "men", "et",
    "har", "om", "vi", "min", "havde", "ham", "hun", "nu", "over", "da", "fra",
    "du", "ud", "sin", "dem", "os", "op", "man", "hans", "hvor", "eller", "hvad",
    "skal", "selv", "her", "alle", "vil", "blev", "kunne", "ind", "når", "være",
}
STOPWORDS = _EN_STOPWORDS | _DA_STOPWORDS


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, accent-normalize, and split into word tokens."""
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = _TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens
