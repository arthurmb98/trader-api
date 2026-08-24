from __future__ import annotations

import argparse
import os
import signal
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trader API — estudo e sinais WIN")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("prepare-data", help="Converte WIN$D MT5 em treino (até 2024) e teste (2025–hoje)")
    sub.add_parser("study", help="Treina no CSV A, testa no CSV B, varre parâmetros")
    sub.add_parser("enrich", help="Agrega períodos e simula contratos compostos nos vencedores")
    sub.add_parser("rerank", help="Reescolhe vencedores com piso de DD < banca e gera o parecer")
    replay = sub.add_parser(
        "replay",
        help="Simula uma janela (padrão: 17–21/08/2026) com a config vencedora M5, sem enviar ordem",
    )
    replay.add_argument("--config", default="best_candles_m5_1000_a")
    replay.add_argument("--from", dest="start", default="2026-08-17")
    replay.add_argument("--to", dest="end", default="2026-08-21")
    replay.add_argument("--csv", default="datasets/mt5_m5_week.csv")
    replay.add_argument("--source", choices=("csv", "mt5", "auto"), default="csv")
    replay.add_argument("--warmup-days", type=int, default=5)
    serve = sub.add_parser("serve", help="Sobe a API FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--debug", action="store_true", help="Log debug + reload")
    sub.add_parser("mt5-check", help="Testa o terminal MT5 (login demo, WIN, order_check, sem enviar ordem)")
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
    if args.cmd == "rerank":
        from trader.study import rerank_saved_study

        rerank_saved_study()
        return
    if args.cmd == "replay":
        from trader.replay import replay_from_args

        replay_from_args(args)
        return
    if args.cmd == "serve":
        import uvicorn

        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            os.setsid()
        except OSError:
            pass
        serve_kw: dict = {
            "host": args.host,
            "port": args.port,
            "reload": args.debug,
            "log_level": "debug" if args.debug else "info",
        }
        if args.debug:
            serve_kw["reload_excludes"] = ["*.json", "web/*", "studies/results/*", ".venv/*"]
        uvicorn.run("trader.api:app", **serve_kw)
        return
    if args.cmd == "mt5-check":
        from trader.mt5_session import mt5_check

        raise SystemExit(mt5_check())
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
