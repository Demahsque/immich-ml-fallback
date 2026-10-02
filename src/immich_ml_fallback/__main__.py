"""Command line: ``python -m immich_ml_fallback {serve,generate,evaluate,explain}``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__

DEFAULT_MODEL_ENV = "CLIP_MODEL"


def _data_dir_default() -> str:
    return os.environ.get("DATA_DIR", "data")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default=_data_dir_default(), help="vectors directory (default: ./data or $DATA_DIR)")


def _add_ml(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ml-url", default="http://localhost:3003", help="Immich ML server (default %(default)s)")
    parser.add_argument(
        "--model",
        default=os.environ.get(DEFAULT_MODEL_ENV),
        required=os.environ.get(DEFAULT_MODEL_ENV) is None,
        help=f"CLIP model name exactly as in Immich settings (or ${DEFAULT_MODEL_ENV})",
    )


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(args.data), host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .generate import ImmichMLClient, MLClientError, build_vocabulary, generate_store
    from .stopwords import load_stopwords

    stopwords = load_stopwords()
    words = build_vocabulary(
        args.languages, args.top_n, stopwords, [Path(p) for p in args.extra_words], not args.no_builtin
    )
    if args.limit:
        words = words[: args.limit]
    print(f"Vocabulary: {len(words)} entries ({sum(' ' in w for w in words)} phrases).", file=sys.stderr)
    if args.dry_run:
        return 0
    client = ImmichMLClient(args.ml_url, args.model)
    try:
        client.ping()
        out = Path(args.data)
        store = generate_store(client, words, out, workers=args.workers, checkpoint=args.checkpoint, rebuild=args.rebuild)
    except MLClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    size_mb = store.vectors.nbytes / 1e6
    print(f"Done: {len(store)} entries, dim {store.dim}, {size_mb:.1f} MB of vectors in {out}/", file=sys.stderr)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .encoder import ApproxEncoder
    from .evaluate import default_queries, evaluate, format_report, queries_from_file
    from .generate import ImmichMLClient, MLClientError
    from .store import STOPWORDS_FILE, StoreError, VectorStore
    from .stopwords import load_stopwords

    try:
        store = VectorStore.load(args.data)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    encoder = ApproxEncoder(store, load_stopwords(Path(args.data) / STOPWORDS_FILE))
    queries = queries_from_file(Path(args.queries)) if args.queries else default_queries()
    client = ImmichMLClient(args.ml_url, args.model)
    try:
        client.ping()
        print(format_report(evaluate(client, encoder, queries)))
    except MLClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .encoder import ApproxEncoder
    from .stopwords import load_stopwords
    from .store import STOPWORDS_FILE, StoreError, VectorStore

    try:
        store = VectorStore.load(args.data)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    analysis = ApproxEncoder(store, load_stopwords(Path(args.data) / STOPWORDS_FILE)).analyse(" ".join(args.query))
    print(f"used:       {list(analysis.used)}\nstop words: {list(analysis.stopped)}\nunknown:    {list(analysis.unknown)}")
    return 0 if analysis.rows else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immich-ml-fallback", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the fallback proxy (NAS side)")
    _add_common(p)
    p.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3003")))
    p.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("generate", help="build the word vectors by querying the Immich ML server (Mac side)")
    _add_common(p)
    _add_ml(p)
    p.add_argument("--languages", nargs="+", default=["fr", "en"], help="wordfreq languages (default: fr en)")
    p.add_argument("--top-n", type=int, default=10000, help="most frequent words per language (0 = none)")
    p.add_argument("--extra-words", action="append", default=[], metavar="FILE", help="your own words/phrases, one per line")
    p.add_argument("--no-builtin", action="store_true", help="skip the bundled extra words and phrases")
    p.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    p.add_argument("--checkpoint", type=int, default=1000, help="save every N entries (default 1000)")
    p.add_argument("--limit", type=int, default=0, help="only the first N entries (quick test)")
    p.add_argument("--rebuild", action="store_true", help="ignore existing vectors and start over")
    p.add_argument("--dry-run", action="store_true", help="only count the vocabulary")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evaluate", help="compare fallback and real embeddings on test queries (Mac side)")
    _add_common(p)
    _add_ml(p)
    p.add_argument("--queries", help="file with one query per line (default: bundled FR/EN list)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("explain", help="show how a query is split into used / ignored / unknown words")
    _add_common(p)
    p.add_argument("query", nargs="+")
    p.set_defaults(func=cmd_explain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
