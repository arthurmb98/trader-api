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

## Ligar o MetaTrader 5

1. Windows com o terminal MT5 aberto e algo trading autorizado.
2. Instale o pacote `MetaTrader5` no Python (não existe no macOS).
3. Em `configs/live.yaml` (ou na config vencedora): `mt5.enabled: true` e o símbolo do vencimento atual (`WIN$` ou `WINQ26`, etc.).
4. `PYTHONPATH=src python -m trader serve`
5. `POST /api/signal` para ver o sinal. `POST /api/orders` envia a ordem com stop e alvo.
6. Comece em **conta simulada**.

## Estrutura

```
configs/          parâmetros (várias configs)
datasets/         CSVs de treino e teste
src/trader/       domínio, ML, backtest, MT5, API
studies/results/  JSON do estudo e modelos .joblib
web/              página de apresentação
```
