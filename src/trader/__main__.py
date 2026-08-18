from __future__ import annotations

import argparse
import sys

import uvicorn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trader API — estudo e sinais WIN")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("study", help="Treina no CSV A, testa no CSV B, varre parâmetros")
    serve = sub.add_parser("serve", help="Sobe a API FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.cmd == "study":
        from trader.study import run_all

        run_all()
        return
    if args.cmd == "serve":
        uvicorn.run("trader.api:app", host=args.host, port=args.port, reload=False)
        return
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
