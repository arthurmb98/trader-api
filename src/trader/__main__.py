from __future__ import annotations

import argparse
import sys

import uvicorn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trader API — estudo e sinais WIN")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("prepare-data", help="Converte WIN$D MT5 em treino (até 2024) e teste (2025–hoje)")
    sub.add_parser("study", help="Treina no CSV A, testa no CSV B, varre parâmetros")
    sub.add_parser("enrich", help="Agrega períodos e simula contratos compostos nos vencedores")
    serve = sub.add_parser("serve", help="Sobe a API FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.cmd == "prepare-data":
        from trader.data import prepare_study_csvs

        prepare_study_csvs()
        return
    if args.cmd == "study":
        from trader.study import run_all

        run_all()
        return
    if args.cmd == "enrich":
        from trader.study import enrich_saved_study

        enrich_saved_study()
        return
    if args.cmd == "serve":
        uvicorn.run("trader.api:app", host=args.host, port=args.port, reload=False)
        return
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
