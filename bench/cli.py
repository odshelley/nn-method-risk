"""CLI for the calibration benchmark."""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("run", "sweep", "aggregate", "figures"):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    if args.cmd == "list":
        print("scenarios: (none registered yet)")
        return 0
    raise NotImplementedError(args.cmd)
