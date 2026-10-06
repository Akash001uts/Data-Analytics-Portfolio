"""Command line entry point: `dap health ...` and `dap sentiment ...`."""

import argparse
import logging
import sys
from pathlib import Path

from dap.common import paths
from dap.common.manifest import ManifestMismatch, load_manifest

NOT_YET = {
    ("health", "build"): "Phase 2",
    ("health", "train"): "Phase 3",
    ("health", "report"): "Phase 4",
    ("sentiment", "fetch"): "Phase 5",
    ("sentiment", "train"): "Phase 5",
    ("sentiment", "report"): "Phase 5",
}


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


def _not_yet(project: str, command: str) -> int:
    phase = NOT_YET[project, command]
    print(f"`dap {project} {command}` is not implemented yet (planned for {phase}).")
    return 2


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
    for name in ("build", "train", "report"):
        p = hcmd.add_parser(name, help=f"not implemented yet ({NOT_YET['health', name]})")
        p.set_defaults(func=lambda _a, n=name: _not_yet("health", n))

    sentiment = projects.add_parser("sentiment", help="sentiment evaluation project")
    scmd = sentiment.add_subparsers(dest="command", required=True)
    for name in ("fetch", "train", "report"):
        p = scmd.add_parser(name, help=f"not implemented yet ({NOT_YET['sentiment', name]})")
        p.set_defaults(func=lambda _a, n=name: _not_yet("sentiment", n))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
