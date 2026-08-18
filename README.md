# Sinal WIN — trader-api

Robô de day trade para o **mini índice (WIN)**: lê o último candle, estima o próximo e sugere compra ou venda, com stop e alvo. Este repositório também gera a **página de apresentação** do estudo de eficácia.

Não é recomendação de investimento. Os números abaixo vêm de dados de 2020 (WINJ20 no treino, WINM20 no teste). Resultado passado não garante resultado futuro.

## O que foi corrigido em relação à versão antiga

- O teste **não treina** o modelo. Treino e teste são arquivos diferentes.
- Qualquer candle do teste que já exista no treino é **removido** (anti-join).
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

## Como gerar de novo o estudo (treino real)

O modelo cabe **só** em `datasets/WINJ20_*.csv`. O P&L roda **só** em `datasets/WINM20_1min.csv` (no 5 minutos, esse arquivo é agregado).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m trader study
```

Isso grava:

- `studies/results/studies.json`
- `web/public/studies.json` (a página lê daqui)
- `configs/best_m1_a.yaml`, `best_m1_b.yaml`, `best_m5_a.yaml`, `best_m5_b.yaml`

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
