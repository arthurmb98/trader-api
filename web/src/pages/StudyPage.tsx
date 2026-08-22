import { useEffect, useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { brl, cn, num, pct } from '@/lib/utils'
import type { BankKey, CaseKey, Parecer, ParecerCaseAvg, PeriodRow, RunSide, StudyFile, TfKey, Winner } from '@/lib/types'

const SIGNAL_CASES: { key: CaseKey; label: string; help: string }[] = [
  {
    key: 'last_candle',
    label: 'Último candle',
    help: 'O modelo lê o candle que acabou de fechar e escolhe compra ou venda.',
  },
  {
    key: 'last_candles',
    label: 'Últimos candles',
    help: 'O mesmo modelo escolhe o lado. Padrões só cancelam se o mercado estiver estranho.',
  },
]
const CASE_HELP = Object.fromEntries(SIGNAL_CASES.map((c) => [c.key, c.help])) as Record<CaseKey, string>
const CASE_LABELS = Object.fromEntries(SIGNAL_CASES.map((c) => [c.key, c.label])) as Record<CaseKey, string>
const TIMEFRAMES: { key: TfKey; label: string }[] = [
  { key: 'm1', label: '1 min' },
  { key: 'm5', label: '5 min' },
]
const BANKS: BankKey[] = ['500', '1000']

function listWinners(data: StudyFile | null, caseKey: CaseKey, bank: BankKey, tf: TfKey): Winner[] {
  const node = data?.winners?.[caseKey]?.[bank]
  if (!node) return []
  if (Array.isArray(node)) {
    return node.filter((winner) => (winner.params.data.timeframe === 'm5' ? 'm5' : 'm1') === tf)
  }
  return node[tf] ?? []
}

function tfOf(row: { timeframe?: string }): TfKey {
  return row.timeframe === 'm1' ? 'm1' : 'm5'
}

function setupLabel(winner: Winner) {
  const risk = winner.params.risk
  const exe = winner.params.execution
  const tf = winner.params.data.timeframe === 'm5' ? '5 min' : '1 min'
  const dir = exe.direction === 'fade' ? 'contra a previsão' : 'seguir'
  const trail = risk.trailing_enabled ? ' · trailing' : ''
  return `${tf} · ${dir}${trail}`
}

function stopGainOf(winner: Winner) {
  return {
    stop: Number(winner.params.risk.stop_points),
    gain: Number(winner.params.risk.gain_points),
  }
}

function stopGainFromLabel(label: string | null | undefined) {
  const match = label?.match(/(\d+)\s*\/\s*(\d+)/)
  if (!match) return null
  return { stop: Number(match[1]), gain: Number(match[2]) }
}

function StopGainMark({
  stop,
  gain,
  className,
}: {
  stop: number
  gain: number
  className?: string
}) {
  if (!Number.isFinite(stop) || !Number.isFinite(gain)) return null
  return (
    <p className={cn('text-[10px] tabular-nums tracking-wide text-muted-foreground/55', className)}>
      stop {stop} · gain {gain}
    </p>
  )
}

function avgMonth(side: RunSide | Winner | undefined) {
  if (!side?.metrics) return 0
  const months = Math.max(side.by_period?.monthly?.length ?? 0, 1)
  return side.metrics.net_pnl / months
}

function lotSide(winner: Winner | undefined, scaled: boolean) {
  if (!winner) return undefined
  if (scaled) return winner.lot_scaled ?? winner.linear ?? winner
  return winner.lot_fixed ?? winner.one_contract ?? winner
}

function LotCol({ label, side, hint }: { label: string; side: RunSide | Winner | undefined; hint?: string }) {
  if (!side?.metrics) return null
  const monthly = avgMonth(side)
  return (
    <div className="rounded-xl border border-border/70 bg-card/40 p-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className={cn('mt-2 font-display text-2xl font-bold', monthly >= 0 ? 'text-gain' : 'text-loss')}>{brl(monthly)}</p>
      <p className="mt-1 text-xs text-muted-foreground">
        /mês · total {brl(side.metrics.net_pnl)} · tombo {pct(side.metrics.max_drawdown_pct)}
        {side.metrics.max_contracts && side.metrics.max_contracts > 1 ? ` · até ${side.metrics.max_contracts} minis` : ''}
      </p>
      {hint ? <p className="mt-2 text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  )
}

function formatDecision(value: unknown) {
  if (value === 'ml_guard') return 'ML + guarda dos últimos candles'
  if (value === 'price_action_ml') return 'padrões definem o lado'
  return 'só o modelo no último candle'
}

function Kpi({
  label,
  value,
  hint,
  positive,
}: {
  label: string
  value: string
  hint?: string
  positive?: boolean | null
}) {
  return (
    <div className="rounded-2xl border border-border bg-elevated/70 p-4">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p
        className={cn(
          'mt-2 font-display text-2xl font-bold',
          positive === true && 'text-gain',
          positive === false && 'text-loss',
        )}
      >
        {value}
      </p>
      {hint ? <p className="mt-1 text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  )
}

function PeriodBars({ title, rows }: { title: string; rows: PeriodRow[] }) {
  const data = rows.length > 120 ? rows.filter((_, i) => i % Math.ceil(rows.length / 120) === 0) : rows
  return (
    <div className="rounded-2xl border border-border bg-elevated/40 p-4">
      <h3 className="font-display font-semibold">{title}</h3>
      <div className="mt-4 h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
            <XAxis dataKey="t" hide />
            <YAxis tick={{ fill: '#a1a1aa', fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
              formatter={(v: number | undefined) => brl(Number(v ?? 0))}
            />
            <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
              {data.map((row) => (
                <Cell key={row.t} fill={row.pnl >= 0 ? '#34d399' : '#fb7185'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function WinnerCard({
  winner,
  title,
  active,
  onClick,
}: {
  winner: Winner
  title: string
  active: boolean
  onClick: () => void
}) {
  const fixed = lotSide(winner, false)
  const scaled = lotSide(winner, true)
  const monthly = avgMonth(fixed)
  const { stop, gain } = stopGainOf(winner)
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'relative rounded-2xl border p-5 text-left transition-colors',
        active ? 'border-primary bg-card' : 'border-border bg-elevated/50 hover:bg-card/80',
      )}
    >
      <StopGainMark stop={stop} gain={gain} className="absolute right-4 top-4" />
      <p className="pr-28 text-sm text-muted-foreground">
        {title}
        {monthly < 0 ? ' · prejuízo no teste' : ''}
      </p>
      <p className="mt-1 font-display text-lg font-semibold">{setupLabel(winner)}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <LotCol label="1 mini" side={fixed} hint="Ranking" />
        <LotCol label="Lote que sobe" side={scaled} hint="Mesmo setup, mais minis" />
      </div>
    </button>
  )
}

function SetupParams({ winner }: { winner: Winner }) {
  const risk = winner.params.risk
  const filters = winner.params.filters
  const exe = winner.params.execution
  const rows: [string, string][] = [
    ['Tempo gráfico', winner.params.data.timeframe === 'm5' ? '5 minutos' : '1 minuto'],
    ['Como decide', formatDecision(exe.decision)],
    ['Direção', exe.direction === 'fade' ? 'contra a previsão' : 'seguir a previsão'],
    ['Stop', `${risk.stop_points} pts`],
    ['Alvo', `${risk.gain_points} pts`],
    ['Trailing', risk.trailing_enabled ? `sim, após ${risk.trailing_trigger_points} pts` : 'não'],
    ['Stop diário', Number(risk.daily_loss_points) > 0 ? `${risk.daily_loss_points} pts` : 'desligado'],
    ['Máx. ops / dia', String(risk.max_trades_per_day)],
    ['Horário-ouro', filters.gold_hours_only ? 'sim' : 'não'],
    ['Gap máximo', filters.max_gap_points == null ? 'sem limite' : `${filters.max_gap_points} pts`],
  ]
  return (
    <div className="rounded-2xl border border-border bg-elevated/50 p-4">
      <h4 className="font-display text-sm font-semibold text-primary">O que muda neste setup</h4>
      <dl className="mt-3 grid gap-2 sm:grid-cols-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-start justify-between gap-4 text-sm">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="text-right font-medium">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-xs text-muted-foreground">
        Igual em todos: sessão 9h15–17h, sem almoço, fill na abertura seguinte, custo R$ 1 por operação.
      </p>
    </div>
  )
}

function CaseCompare({
  parecer,
  caseLabels,
  caseList,
  lookback,
  onPick,
}: {
  parecer: Parecer
  caseLabels: Record<string, string>
  caseList: CaseKey[]
  lookback: { m1: number; m5: number }
  onPick: (key: CaseKey, bank?: BankKey, tf?: TfKey) => void
}) {
  const rows = parecer.by_case?.length
    ? parecer.by_case
    : parecer.monthly.reduce<ParecerCaseAvg[]>((acc, row) => {
        if (acc.some((item) => item.case === row.case && item.bank === row.bank && tfOf(item) === tfOf(row))) return acc
        acc.push(row)
        return acc
      }, [])

  return (
    <section id="parecer" className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
      <p className="text-sm uppercase tracking-[0.2em] text-primary">Comparar</p>
      <h2 className="mt-2 font-display text-3xl font-bold">{parecer.headline}</h2>
      <p className="mt-3 max-w-3xl text-muted-foreground">
        Número grande = ganho médio mensal do melhor setup de cada caso, banca e tempo gráfico.{' '}
        {parecer.n_months_note} Acerto do ML: {pct(parecer.ml_hit.m1)} no 1 min ({lookback.m1} candles de contexto na
        guarda) e {pct(parecer.ml_hit.m5)} no 5 min ({lookback.m5} candles). {parecer.dd_floor}
      </p>
      <div className="mt-8 grid gap-4 lg:grid-cols-2">
        {caseList.map((key) => (
          <div key={key} className="rounded-2xl border border-border bg-elevated/50 p-5">
            <p className="font-display text-xl font-semibold">{CASE_LABELS[key] ?? caseLabels[key] ?? key}</p>
            <p className="mt-2 text-sm text-muted-foreground">{CASE_HELP[key]}</p>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              {BANKS.map((bank) => (
                <div key={`${key}-${bank}`} className="rounded-xl border border-border/70 bg-card/40 p-3">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">{brl(Number(bank))}</p>
                  <div className="mt-3 grid gap-2">
                    {TIMEFRAMES.map((tf) => {
                      const row = rows.find((item) => item.case === key && String(item.bank) === bank && tfOf(item) === tf.key)
                      const sg = stopGainFromLabel(row?.label)
                      return (
                        <button
                          key={tf.key}
                          type="button"
                          onClick={() => onPick(key, bank, tf.key)}
                          className="relative rounded-lg border border-border/60 bg-background/40 p-2 text-left transition-colors hover:border-primary"
                        >
                          {sg ? <StopGainMark stop={sg.stop} gain={sg.gain} className="absolute right-2 top-2" /> : null}
                          <p className="pr-24 text-[11px] uppercase tracking-wide text-muted-foreground">{tf.label}</p>
                          {row?.avg_fixed == null ? (
                            <p className="mt-1 text-xs text-muted-foreground">Sem setup neste recorte.</p>
                          ) : (
                            <>
                              <p className={cn('mt-1 font-display text-xl font-bold', row.avg_fixed >= 0 ? 'text-gain' : 'text-loss')}>
                                {brl(row.avg_fixed)}
                              </p>
                              <p className="text-[11px] text-muted-foreground">1 mini / mês</p>
                              <p className={cn('mt-1 text-xs font-medium', (row.avg_scaled ?? 0) >= 0 ? 'text-gain' : 'text-loss')}>
                                {brl(row.avg_scaled ?? 0)}{' '}
                                <span className="font-normal text-muted-foreground">lote que sobe</span>
                              </p>
                            </>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <ul className="mt-8 space-y-2 text-sm text-foreground/90">
        {parecer.strategy.map((item) => (
          <li key={item}>· {item}</li>
        ))}
      </ul>
    </section>
  )
}

export function StudyPage() {
  const [data, setData] = useState<StudyFile | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [caseKey, setCaseKey] = useState<CaseKey>('last_candle')
  const [bank, setBank] = useState<BankKey>('1000')
  const [tf, setTf] = useState<TfKey>('m5')
  const [pick, setPick] = useState(0)
  const [chartScaled, setChartScaled] = useState(false)
  const [showIntra, setShowIntra] = useState(false)

  useEffect(() => {
    const load = async () => {
      const bust = `v=${Date.now()}`
      const urls = [`/studies.json?${bust}`, `/api/studies?${bust}`]
      for (const url of urls) {
        try {
          const res = await fetch(url, { cache: 'no-store' })
          if (!res.ok) continue
          const json = (await res.json()) as StudyFile
          const winnerKeys = Object.keys(json.winners ?? {})
          if (winnerKeys.includes('one_contract') && !winnerKeys.includes('last_candle')) {
            continue
          }
          if (winnerKeys.includes('last_candles_ml') && !winnerKeys.includes('last_candles')) {
            continue
          }
          setData(json)
          setCaseKey('last_candle')
          const firstBank = String(json.banks?.[1] ?? json.banks?.[0] ?? 1000) as BankKey
          setBank(firstBank)
          setTf('m5')
          setChartScaled(false)
          return
        } catch {
          /* try next */
        }
      }
      setError('Estudo ainda não disponível. Rode python -m trader study.')
    }
    void load()
  }, [])

  const caseWinners = listWinners(data, caseKey, bank, tf)
  const winner = caseWinners[Math.min(pick, Math.max(caseWinners.length - 1, 0))]
  const side = lotSide(winner, chartScaled)
  const hourly = useMemo(() => {
    if (!side) return []
    return Object.entries(side.metrics.hourly)
      .map(([hour, v]) => ({ hour: `${hour}h`, ...v }))
      .sort((a, b) => Number(a.hour.replace('h', '')) - Number(b.hour.replace('h', '')))
  }, [side])

  if (error) {
    return (
      <main className="mx-auto max-w-xl px-5 py-24 text-center">
        <AlertTriangle className="mx-auto size-10 text-pink" />
        <p className="mt-4 text-muted-foreground">{error}</p>
      </main>
    )
  }
  if (!data) {
    return (
      <main className="grid min-h-dvh place-items-center">
        <p className="animate-fade-in text-muted-foreground">Carregando o estudo…</p>
      </main>
    )
  }

  const caseList = SIGNAL_CASES.map((item) => item.key)
  const lookback = data.lookback ?? { m1: 10, m5: 5 }
  const leak = data.timeframes.m1.leakage
  const m = side?.metrics
  const periods = side?.by_period ?? winner?.by_period
  const bankNum = Number(bank)
  const scaleHint =
    bankNum === 500
      ? 'Lote que sobe: 1 mini em R$ 500, 2 em R$ 1.000, teto 16.'
      : 'Lote que sobe: 1 mini em R$ 1.000, 2 em R$ 2.000, teto 16.'

  const pickCase = (key: CaseKey, nextBank?: BankKey, nextTf?: TfKey) => {
    setCaseKey(key)
    if (nextBank) setBank(nextBank)
    if (nextTf) setTf(nextTf)
    setPick(0)
    setChartScaled(false)
    document.getElementById('setups')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <header className="relative mx-auto flex max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
        <p className="font-display text-lg font-bold">Sinal WIN</p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <a href="/live">Ao vivo</a>
          </Button>
          <Button variant="outline" size="sm" onClick={() => document.getElementById('parecer')?.scrollIntoView()}>
            Comparar
          </Button>
          <Button variant="outline" size="sm" onClick={() => document.getElementById('setups')?.scrollIntoView()}>
            Setups
          </Button>
          <Button size="sm" onClick={() => document.getElementById('mt5')?.scrollIntoView()}>
            Até o MT5
          </Button>
        </div>
      </header>

      <section className="relative mx-auto max-w-6xl px-5 pb-12 pt-10 sm:px-8">
        <p className="animate-fade-up text-sm uppercase tracking-[0.2em] text-primary">Estudo para sócios</p>
        <h1 className="mt-4 max-w-3xl animate-fade-up font-display text-4xl font-bold leading-tight sm:text-5xl [animation-delay:80ms]">
          O robô lê o último candle e sugere compra ou venda.
        </h1>
        <p className="mt-5 max-w-2xl animate-fade-up text-lg text-muted-foreground [animation-delay:140ms]">
          Mini índice WIN$ contínuo. Treino até dez/2024, teste jan/2025–ago/2026. Sinal no fechamento, execução na
          abertura do próximo candle. {leak.n_removed} candles repetidos saíram do teste.
        </p>
        <div className="mt-8 grid gap-4 md:grid-cols-2">
          {SIGNAL_CASES.map((item) => (
            <div key={item.key} className="rounded-2xl border border-border bg-elevated/60 p-5">
              <p className="font-display text-lg font-semibold">{item.label}</p>
              <p className="mt-2 text-sm text-muted-foreground">{item.help}</p>
              {item.key === 'last_candles' ? (
                <p className="mt-3 text-xs text-muted-foreground">
                  Janela da guarda: {lookback.m1} candles no 1 min, {lookback.m5} no 5 min.
                </p>
              ) : null}
            </div>
          ))}
        </div>
        <p className="mt-6 text-sm text-muted-foreground">
          Banca de R$ 500 ou R$ 1.000. Tempo gráfico de 1 min ou 5 min, cada um com os dois melhores setups. Lote: 1
          mini o tempo todo, ou +1 a cada múltiplo da banca (teto 16). Stop diário em pontos é o mesmo nas duas bancas.
          Ponto = {brl(data.instrument.point_value)}.
        </p>
      </section>

      {data.parecer ? (
        <CaseCompare
          parecer={data.parecer}
          caseLabels={CASE_LABELS}
          caseList={caseList}
          lookback={lookback}
          onPick={pickCase}
        />
      ) : null}

      <section id="setups" className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <h2 className="font-display text-3xl font-bold">Melhor e segundo melhor</h2>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Filtre por caso, banca e tempo gráfico. Cada combinação tem os dois melhores setups. 1 mini é o ranking; lote
          que sobe é o mesmo setup. {scaleHint} Gerado em {new Date(data.generated_at).toLocaleString('pt-BR')}.
        </p>

        <div className="mt-6 grid gap-4 rounded-2xl border border-border bg-elevated/40 p-4 sm:grid-cols-3">
          <div>
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Caso</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {SIGNAL_CASES.map((item) => (
                <Button
                  key={item.key}
                  variant={caseKey === item.key ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {
                    setCaseKey(item.key)
                    setPick(0)
                    setChartScaled(false)
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </div>
            <p className="mt-2 text-xs text-muted-foreground">{CASE_HELP[caseKey]}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Banca inicial</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {(data.banks.length ? data.banks : [500, 1000]).map((value) => {
                const key = String(value) as BankKey
                return (
                  <Button
                    key={key}
                    variant={bank === key ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => {
                      setBank(key)
                      setPick(0)
                      setChartScaled(false)
                    }}
                  >
                    {brl(value)}
                  </Button>
                )
              })}
            </div>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Tempo gráfico</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {TIMEFRAMES.map((item) => (
                <Button
                  key={item.key}
                  variant={tf === item.key ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {
                    setTf(item.key)
                    setPick(0)
                    setChartScaled(false)
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              {tf === 'm1' ? 'Candle de 1 minuto.' : 'Candle de 5 minutos.'}
            </p>
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {caseWinners.map((w, i) => (
            <WinnerCard
              key={w.params.name || w.name}
              winner={w}
              title={i === 0 ? 'melhor' : 'segundo melhor'}
              active={pick === i}
              onClick={() => setPick(i)}
            />
          ))}
        </div>
        {!winner ? (
          <p className="mt-6 rounded-2xl border border-border bg-elevated/50 p-5 text-sm text-muted-foreground">
            Nenhum setup rodou neste caso, banca e tempo gráfico.
          </p>
        ) : null}

        {m && winner && side ? (
          <>
            <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Kpi label="P&L no teste (20 meses)" value={brl(m.net_pnl)} positive={m.net_pnl >= 0} />
              <Kpi label="Maior tombo" value={brl(m.max_drawdown)} hint={pct(m.max_drawdown_pct)} />
              <Kpi label="Acerto" value={pct(m.win_rate)} hint={`${m.n_wins} gains · ${m.n_losses} stops`} />
              <Kpi label="Operações" value={String(m.n_trades)} hint={`fator ${num(m.profit_factor, 2)}`} />
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-5">
              <div className="rounded-2xl border border-border bg-elevated/40 p-4 lg:col-span-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-display font-semibold">Curva da banca</h3>
                  <Button variant="outline" size="sm" onClick={() => setChartScaled((v) => !v)}>
                    {chartScaled ? 'Ver curva em 1 mini' : 'Ver curva do lote que sobe'}
                  </Button>
                </div>
                {chartScaled ? (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Mesmo setup, lote crescente. Tombo pode passar da banca — não entra no ranking.
                  </p>
                ) : null}
                <div className="mt-4 h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={m.equity}>
                      <defs>
                        <linearGradient id="bank" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#058ef2" stopOpacity={0.7} />
                          <stop offset="100%" stopColor="#9f2db3" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
                      <XAxis dataKey="t" hide />
                      <YAxis tick={{ fill: '#a1a1aa', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
                        formatter={(v: number | undefined) => brl(Number(v ?? 0))}
                      />
                      <Area type="monotone" dataKey="bank" stroke="#058ef2" fill="url(#bank)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="rounded-2xl border border-border bg-elevated/40 p-4 lg:col-span-2">
                <h3 className="font-display font-semibold">Por horário</h3>
                <div className="mt-4 h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={hourly}>
                      <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
                      <XAxis dataKey="hour" tick={{ fill: '#a1a1aa', fontSize: 12 }} />
                      <YAxis tick={{ fill: '#a1a1aa', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
                        formatter={(v: number | undefined) => brl(Number(v ?? 0))}
                      />
                      <Bar dataKey="pnl" radius={[6, 6, 0, 0]}>
                        {hourly.map((row) => (
                          <Cell key={row.hour} fill={row.pnl >= 0 ? '#34d399' : '#fb7185'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {periods ? (
              <div className="mt-8 space-y-4">
                <PeriodBars title="Resultado mensal" rows={periods.monthly} />
                <Button variant="outline" size="sm" onClick={() => setShowIntra((v) => !v)}>
                  {showIntra ? 'Ocultar dias e semanas' : 'Ver dias e semanas'}
                </Button>
                {showIntra ? (
                  <div className="grid gap-4 lg:grid-cols-2">
                    <PeriodBars title="Diário" rows={periods.daily} />
                    <PeriodBars title="Semanal" rows={periods.weekly} />
                  </div>
                ) : null}
              </div>
            ) : null}

            <div className="mt-8">
              <SetupParams winner={winner} />
            </div>
          </>
        ) : null}
      </section>

      <section id="mt5" className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <h2 className="font-display text-3xl font-bold">Do estudo ao MetaTrader 5</h2>
        <ol className="mt-6 space-y-3">
          {data.mt5.steps.map((step, i) => (
            <li key={step} className="flex gap-4 rounded-2xl border border-border bg-elevated/50 p-4">
              <span className="font-display text-primary">{i + 1}</span>
              <p>{step}</p>
            </li>
          ))}
        </ol>
        {data.insights.improve.length ? (
          <ul className="mt-6 space-y-2 text-sm text-muted-foreground">
            {data.insights.improve.map((item) => (
              <li key={item}>· {item}</li>
            ))}
          </ul>
        ) : null}
        <p className="mt-6 text-sm text-muted-foreground">{data.disclaimer}</p>
        <p className="mt-2 text-xs text-muted-foreground">Gerado em {new Date(data.generated_at).toLocaleString('pt-BR')}</p>
      </section>

      <footer className="border-t border-border py-8 text-center text-sm text-muted-foreground">
        <p>Sinal WIN · estudo educacional · não é recomendação de investimento</p>
        <a
          href="https://koletivo-hub.vercel.app"
          target="_blank"
          rel="noreferrer"
          className="mt-5 inline-flex flex-col items-center gap-3 text-foreground hover:text-primary"
        >
          <img src="/brand/logo-branco.png" alt="Koletivo Hub" className="h-10 w-auto object-contain" />
          <span className="font-display text-sm font-semibold">Desenvolvido por Koletivo Hub</span>
        </a>
        <p className="mt-3 text-xs text-muted-foreground/70">
          © {new Date().getFullYear()}. Todos os direitos reservados a Koletivo Hub.
        </p>
      </footer>
    </div>
  )
}
