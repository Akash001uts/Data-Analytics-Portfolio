"""Command line entry point: `dap health ...` and `dap sentiment ...`."""

import argparse
import logging
import sys
from pathlib import Path

from dap.common import paths
from dap.common.manifest import ManifestMismatch, load_manifest


def _health_fetch(args: argparse.Namespace) -> int:
    from dap.health.fetch import fetch_all

    manifest = load_manifest(args.manifest or paths.manifest_path())
    raw = Path(args.raw_dir) if args.raw_dir else paths.raw_dir()
    try:
        done = fetch_all(manifest, raw, only=args.only, force=args.force)
    except ManifestMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for sid, path in done.items():
        print(f"ok  {sid:32} {path}")
    return 0


def _health_build(args: argparse.Namespace) -> int:
    from dap.health.build import build

    try:
        out = build(raw_dir=Path(args.raw_dir) if args.raw_dir else paths.raw_dir())
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out['table']}")
    print(f"wrote {out['summary']}")
    s = out["stats"]
    print(f"{s['sa3_count']} SA3s, {s['sa3_with_target']} with a {s['target_year']} target")
    return 0


def _health_train(args: argparse.Namespace) -> int:
    from dap.health.train import train

    try:
        out = train()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out['results']}")
    print(f"wrote {out['residuals']}")
    for f in out["figures"]:
        print(f"wrote {f}")
    models = out["stats"]["models"]
    for m in models.values():
        sp, rd = m["spatial"]["r2"]["mean"], m["random"]["r2"]["mean"]
        print(f"{m['label']:26} R2 grouped by SA4 {sp:6.3f}   random {rd:6.3f}")
    return 0


def _health_report(args: argparse.Namespace) -> int:
    from dap.health.report import report

    try:
        out = report()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def _sentiment_fetch(args: argparse.Namespace) -> int:
    from dap.health.fetch import fetch_all

    manifest = load_manifest(paths.sentiment_manifest_path())
    try:
        done = fetch_all(manifest, paths.raw_dir(), force=args.force)
    except ManifestMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for sid, path in done.items():
        print(f"ok  {sid:32} {path}")
    return 0


def _sentiment_transformer(args: argparse.Namespace) -> int:
    from dap.sentiment.data import load_reviews
    from dap.sentiment.transformer import predict

    try:
        reviews, _ = load_reviews()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = predict(reviews[reviews.in_eval][["review_id", "text"]], limit=args.limit)
    print(f"wrote {out}")
    return 0


def _sentiment_train(args: argparse.Namespace) -> int:
    from dap.sentiment.train import train

    try:
        out = train()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out['results']}")
    for f in out["figures"]:
        print(f"wrote {f}")
    for m in out["stats"]["models"].values():
        f1 = m["macro_f1"]
        ci = f"{f1['ci_low']:.3f} to {f1['ci_high']:.3f}"
        print(f"{m['label']:34} macro-F1 {f1['estimate']:.3f} ({ci})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dap", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    projects = parser.add_subparsers(dest="project", required=True)

    health = projects.add_parser("health", help="avoidable hospital admissions project")
    hcmd = health.add_subparsers(dest="command", required=True)
    fetch = hcmd.add_parser("fetch", help="download raw data and check it against the manifest")
    fetch.add_argument("--manifest", type=Path, help="manifest file (default: data/manifest.yaml)")
    fetch.add_argument("--raw-dir", help="where raw files go (default: data/raw)")
    fetch.add_argument("--only", nargs="+", metavar="ID", help="fetch only these source ids")
    fetch.add_argument("--force", action="store_true", help="download again even if verified")
    fetch.set_defaults(func=_health_fetch)
    build = hcmd.add_parser("build", help="clean and join the raw data into one SA3 table")
    build.add_argument("--raw-dir", help="where raw files are (default: data/raw)")
    build.set_defaults(func=_health_build)
    train = hcmd.add_parser("train", help="spatial statistics, models and reports/results.json")
    train.set_defaults(func=_health_train)
    report = hcmd.add_parser("report", help="the interactive map, reports/map/index.html")
    report.set_defaults(func=_health_report)

    sentiment = projects.add_parser("sentiment", help="sentiment evaluation project")
    scmd = sentiment.add_subparsers(dest="command", required=True)
    sfetch = scmd.add_parser(
        "fetch", help="download the reviews and check them against the manifest"
    )
    sfetch.add_argument("--force", action="store_true", help="download again even if verified")
    sfetch.set_defaults(func=_sentiment_fetch)
    trans = scmd.add_parser(
        "transformer", help="run RoBERTa on the evaluation sample and cache it (needs --group nlp)"
    )
    trans.add_argument("--limit", type=int, help="only score this many more reviews")
    trans.set_defaults(func=_sentiment_transformer)
    strain = scmd.add_parser("train", help="fit the baselines, evaluate everything, write results")
    strain.set_defaults(func=_sentiment_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
