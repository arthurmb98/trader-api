# Sinal WIN — trader-api

Robô de day trade para o **mini índice (WIN)**: lê o último candle, estima o próximo e sugere compra ou venda, com stop e alvo. Este repositório também gera a **página de apresentação** do estudo de eficácia.

Não é recomendação de investimento. Os números vêm do contínuo **WIN$**: treino de ago/2021 a 31/12/2024 e teste de 02/01/2025 a ago/2026. Resultado passado não garante resultado futuro.

## O que foi corrigido em relação à versão antiga

- O teste **não treina** o modelo. Treino e teste são arquivos diferentes, com parede em 2025.
- Qualquer candle do teste que já exista no treino (mesmo ativo + timestamp) é **removido**.
- Os quatro modelos (mínimo/fechamento) não estão mais trocados.
- High/Low do MetaTrader 5 não estão mais invertidos.
- A API agora **pode enviar ordem** (`order_send` com SL/TP), não só ler o candle.
- Banca, contratos, valor do ponto, stop, gain e filtros ficam em `configs/*.yaml`.

## Como o sócio vê o estudo

Use Node 20 (há um `.nvmrc` em `web/`).

```bash
cd web
nvm use
npm install
npm run dev
```

Abra o endereço que o Vite mostrar (em geral http://127.0.0.1:5173). A página usa as cores e as fontes do Koletivo (Syne + Outfit).

### Publicar só a apresentação no Vercel

O `vercel.json` na raiz manda o Vercel instalar e buildar **apenas** `web/`. A API Python não vai para a nuvem.

Depois do push, importe o repositório no [Vercel](https://vercel.com) (framework detectado pelo arquivo de config) ou rode `npx vercel --prod` na raiz.

Para API + página juntas (depois do `npm run build` em `web/`):

```bash
PYTHONPATH=src .venv/bin/python -m trader serve
```

## Como gerar de novo o estudo

1. Coloque os dumps MT5 em `datasets/WIN$D(M1).csv` e `datasets/WIN$D(M5).csv`.
2. Separe treino (até 2024) e teste (2025–hoje):

```bash
PYTHONPATH=src python -m trader prepare-data
```

Isso grava `datasets/WIN_1min_train.csv`, `WIN_1min_test.csv`, `WIN_5min_train.csv` e `WIN_5min_test.csv`. O 5 minutos é nativo (não é resample do 1 min).

O modelo cabe **só** no treino (até 2024). O P&L roda **só** no teste (2025–hoje). O ranking usa 1 contrato; `python -m trader enrich` (já incluso em `study`) agrega dia/semana/mês e simula contratos em potência de 2.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m trader study
```

Isso grava:

- `studies/results/studies.json`
- `web/public/studies.json` (a página lê daqui)
- `configs/best_m1_500_a.yaml` … `best_m5_5000_b.yaml` (2 por banca e por timeframe)

## Replay da última semana (sem enviar ordem)

Usa a melhor setup de **últimos candles · 5 min · R$ 1000** (`best_candles_m5_1000_a`): ML + guarda de mercado estranho, follow 100/200, trailing, horário de ouro, 1 mini.

O Python `MetaTrader5` **não roda no macOS**. Neste Mac, o terminal só exporta o histórico; o replay roda no Python.

1. No MT5, logado na demo com **WIN$** (ou o vencimento) no Market Watch.
2. Gráfico M5 → clique direito → Exportar / Salvar como CSV, de ~10/08 até 21/08/2026.
3. Salve em `datasets/mt5_m5_week.csv`.
4. Alternativa: compile `mt5/ExportM5Week.mq5` no MetaEditor, rode no gráfico M5 e copie o CSV de `MQL5/Files` para `datasets/mt5_m5_week.csv`.

```bash
PYTHONPATH=src python -m trader replay \
  --config best_candles_m5_1000_a \
  --from 2026-08-17 --to 2026-08-21 \
  --csv datasets/mt5_m5_week.csv
```

No **Windows**, com o terminal aberto, dá para pular o CSV:

```bash
PYTHONPATH=src python -m trader replay --source mt5
```

Se `datasets/mt5_m5_week.csv` não existir, o comando avisa e tenta `datasets/WIN_5min_test.csv` (pode faltar o fim da semana).

## Replay e ao vivo (front)

- `/replay` (a antiga “Ao vivo”) simula janelas de CSV. Não envia ordem.
- `/ao-vivo` opera de verdade no MT5 local, setup `best_candles_m5_1000_a` (M5, R$ 1.000, 1 mini).

## Ligar o MetaTrader 5 (conta demo)

WIN na B3 **não existe** na demo genérica da MetaQuotes. A API só se anexa ao terminal já aberto.

1. Abra conta em [cadastro.clear.com.br](https://cadastro.clear.com.br) (ou use XP/Rico/Clear se já for cliente).
2. Contrate **MetaTrader 5 Demo** em Plataformas (~R$ 9,90/mês). O e-mail com login pode levar até 24h.
3. Instale o MT5 **DEMO da Clear**. Arquivo → Abrir uma conta → CLEAR CTVM → CLEAR DEMO.
4. Market Watch: vencimento atual (`WINV26` em ago/2026). AutoTrading verde.
5. Windows:

```bash
pip install -r requirements-windows.txt
set PYTHONPATH=src
python -m trader mt5-check
python -m trader serve
```

Em outro terminal: `cd web && npm run dev`. Abra http://127.0.0.1:5173/ao-vivo.

O motor arma sozinho com a API. Fora do ouro (09:15–11:00 e 14:30–17:00) ele espera. **Conta real é recusada.** Deixe o Windows ligado, o MT5 aberto e a API rodando.

No Ao vivo o seletor **Fonte** tem MetaTrader 5 e Stream (paper). Se o MT5 cair, troque na mão para Stream: candles via `POST /api/realtime/candles`, `WIN_STREAM_URL` ou `datasets/win_stream.json`. Ordens no Stream são simuladas.

Opcional: copie `.env.example` para `.env` com `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER`. Não commite senha.

A apresentação no Vercel continua só com o estudo e o replay paper. `/api/realtime` no host devolve 400.

## Estrutura

```
configs/          parâmetros (várias configs)
datasets/         CSVs de treino e teste
src/trader/       domínio, ML, backtest, MT5, API
studies/results/  JSON do estudo e modelos .joblib
web/              página de apresentação
```
