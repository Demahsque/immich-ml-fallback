"""Function words that carry no visual meaning, ignored in queries (French + English).

Deliberately absent: words that are also useful nouns in one of the two languages
("car", "son", "or", "été", "photo"...).
"""

from __future__ import annotations

from pathlib import Path

from .text import tokenize

_FR = """
le la les l un une des du de d en et ou où ni ne n pas plus que qu qui quoi dont quand comme si mais donc
à a au aux ce cet cette ces c ça ca cela celui celle ceux celles ci là y se s sa ses mon ma mes ton ta tes t
notre nos votre vos leur leurs je j tu te il elle on nous vous ils elles me m moi toi lui eux
ai as avons avez ont avais avait avions aviez avaient eu avoir est es sommes êtes etes sont étais était
etais etait étions étaient etaient suis être etre fait faire
sur sous dans par pour avec sans chez vers entre contre depuis pendant avant après apres durant selon sauf
très tres trop aussi encore déjà deja alors ainsi puis enfin tout tous toute toutes même meme autre autres
quel quelle quels quelles quelque quelques plusieurs chaque certain certains certaines aucun aucune
lorsque puisque jusqu lorsqu puisqu quoiqu
"""

_EN = """
a an the of in on at to for from by with without within and or but if then than as is are was were be been
being am do does did done have has had having i you he she it we they me him her us them my your his its our
their this that these those there here what which who whom whose when where why how not no nor so too very
can could will would should shall may might must just about into onto over under up down out off again more
most some any each every other such own same only also both few all s t d ll re ve m
"""

FR_STOPWORDS: frozenset[str] = frozenset(_FR.split())
EN_STOPWORDS: frozenset[str] = frozenset(_EN.split())
BUILTIN_STOPWORDS: frozenset[str] = FR_STOPWORDS | EN_STOPWORDS


def load_stopwords(extra_file: Path | None = None) -> frozenset[str]:
    """Built-in stop words plus the optional extra file (one word per line, '#' comments)."""
    words = set(BUILTIN_STOPWORDS)
    if extra_file is not None and extra_file.is_file():
        for line in extra_file.read_text(encoding="utf-8").splitlines():
            words.update(tokenize(line.split("#", 1)[0]))
    return frozenset(words)
