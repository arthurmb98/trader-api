import { useEffect, useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Button } from '@/components/ui/button'
import { brl, cn, pct } from '@/lib/utils'

type SignalSnap = {
  side: string
  entry: number
  stop: number
  take: number
  reason: string
  predicted?: { abertura: number; maximo: number; minimo: number; fechamento: number } | null
}

type TradeSnap = {
  side: string
  entry_time: string
  exit_time: string
  entry: number
  exit: number
  points: number
  pnl: number
  result: string
  reason: string
  contracts?: number
}

type CaseKey = 'last_candle' | 'last_candles'
type TfKey = 'm1' | 'm5'

type LiveSnap = {
  running: boolean
  done: boolean
  error: string | null
  config: string
  case?: CaseKey | string
  timeframe?: TfKey | string
  source: string
  interval_sec: number
  window_start?: string | null
  window_end?: string | null
  start?: string | null
  end?: string | null
  last_tick: string | null
  last_bar_time: string | null
  cursor: number
  n_bars: number
  initial_bank: number
  bank: number
  net_pnl: number
  today_pnl: number
  avg_daily: number
  n_days: number
  n_trades: number
  n_wins: number
  win_rate: number
  max_drawdown: number
  max_drawdown_pct: number
  contracts: number
  max_contracts: number
  signal: SignalSnap | null
  position: {
    side: string
    entry: number
    stop: number
    take: number
    time: string | null
    reason?: string
    contracts?: number
  } | null
  trades: TradeSnap[]
  equity: { t: string; bank: number }[]
  daily: { t: string; pnl: number }[]
  signals: (SignalSnap & { t: string })[]
  candles: { t: string; close: number; open: number; high: number; low: number }[]
  periods?: PeriodStats
}

type PeriodLevel = 'daily' | 'weekly' | 'monthly' | 'quarterly'

type PeriodAvg = {
  avg: number
  avg_gain: number
  avg_loss: number
  n: number
  n_gain: number
  n_loss: number
}

type PeriodStats = {
  window_days: number
  levels: PeriodLevel[]
  series: Partial<Record<PeriodLevel, { t: string; pnl: number }[]>>
  avg: Partial<Record<PeriodLevel, PeriodAvg>>
}

type LiveMeta = {
  banks: number[]
  cases: { key: CaseKey; label: string }[]
  timeframes: { key: TfKey; label: string }[]
  timeframe: TfKey
  min_date: string
  max_date: string
  default_start: string
  default_end: string
  max_span_months: number
}

const EMPTY: LiveSnap = {
  running: false,
  done: false,
  error: null,
  config: 'best_candles_m5_1000_a',
  source: 'paper',
  interval_sec: 1,
  last_tick: null,
  last_bar_time: null,
  cursor: 0,
  n_bars: 0,
  initial_bank: 1000,
  bank: 1000,
  net_pnl: 0,
  today_pnl: 0,
  avg_daily: 0,
  n_days: 0,
  n_trades: 0,
  n_wins: 0,
  win_rate: 0,
  max_drawdown: 0,
  max_drawdown_pct: 0,
  contracts: 1,
  max_contracts: 1,
  signal: null,
  position: null,
  trades: [],
  equity: [],
  daily: [],
  signals: [],
  candles: [],
  periods: { window_days: 0, levels: ['daily'], series: { daily: [] }, avg: {} },
}

function clock(value: string | null | undefined) {
  if (!value) return '—'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return value.replace('T', ' ').slice(0, 19)
  return d.toLocaleString('pt-BR')
}

function dayLabel(value: string | null | undefined) {
  if (!value) return '—'
  const [y, m, d] = value.slice(0, 10).split('-')
  if (!y || !m || !d) return value.slice(0, 10)
  return `${d}/${m}/${y}`
}

function addMonths(iso: string, months: number) {
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number)
  const dt = new Date(Date.UTC(y, m - 1 + months, 1))
  const last = new Date(Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth() + 1, 0)).getUTCDate()
  dt.setUTCDate(Math.min(d, last))
  return dt.toISOString().slice(0, 10)
}

function windowTooLong(start: string, end: string) {
  if (!start || !end) return false
  return end > addMonths(start, 3)
}

function minIso(a: string, b: string) {
  if (!a) return b
  if (!b) return a
  return a < b ? a : b
}

function maxIso(a: string, b: string) {
  if (!a) return b
  if (!b) return a
  return a > b ? a : b
}

const RANGE_MIN = '2025-01-01'
  last_candle: 'Último candle',
  last_candles: 'Últimos candles',
}
const TF_LABEL: Record<TfKey, string> = { m1: '1 min', m5: '5 min' }
const FALLBACK_BANKS = [500, 1000, 2000, 3000, 5000, 10000, 15000]
const selectClass = 'h-9 rounded-lg border border-border bg-elevated/60 px-3 text-sm'
const PERIOD_LABEL: Record<PeriodLevel, string> = {
  daily: 'Diário',
  weekly: 'Semanal',
  monthly: 'Mensal',
  quarterly: 'Trimestral',
}
const PERIOD_UNIT: Record<PeriodLevel, string> = {
  daily: 'dia',
  weekly: 'semana',
  monthly: 'mês',
  quarterly: 'trimestre',
}

function asCase(value: string | undefined | null): CaseKey {
  return value === 'last_candle' ? 'last_candle' : 'last_candles'
}

function asTf(value: string | undefined | null): TfKey {
  return value === 'm1' ? 'm1' : 'm5'
}

function clientLevels(start: string, end: string): PeriodLevel[] {
  if (!start || !end) return ['daily']
  const from = new Date(`${start}T00:00:00`)
  const to = new Date(`${end}T00:00:00`)
  const days = Math.round((to.getTime() - from.getTime()) / 86_400_000) + 1
  const levels: PeriodLevel[] = ['daily']
  if (days >= 7) levels.push('weekly')
  if (days >= 30) levels.push('monthly')
  if (days >= 90) levels.push('quarterly')
  return levels
}

function periodTick(level: PeriodLevel, t: string) {
  if (level === 'daily') return t.slice(5)
  if (level === 'monthly') return t
  return t.replace(/^\d{4}-/, '')
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
          'mt-2 font-display text-2xl font-bold tabular-nums',
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

function PeriodAvgCard({ level, stats }: { level: PeriodLevel; stats?: PeriodAvg }) {
  const avg = stats?.avg ?? 0
  const gain = stats?.avg_gain ?? 0
  const loss = stats?.avg_loss ?? 0
  const unit = PERIOD_UNIT[level]
  return (
    <div className="rounded-2xl border border-border bg-elevated/70 p-4">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">Média {PERIOD_LABEL[level].toLowerCase()}</p>
      <p
        className={cn(
          'mt-2 font-display text-2xl font-bold tabular-nums',
          avg > 0 && 'text-gain',
          avg < 0 && 'text-loss',
        )}
      >
        {brl(avg)}
      </p>
      <p className="mt-2 text-xs text-muted-foreground">
        Ganhos {brl(gain)}
        {stats?.n_gain ? ` · ${stats.n_gain} ${unit}(s)` : ''}
      </p>
      <p className="text-xs text-muted-foreground">
        Perdas {brl(loss)}
        {stats?.n_loss ? ` · ${stats.n_loss} ${unit}(s)` : ''}
      </p>
    </div>
  )
}

function PeriodChart({ title, rows }: { title: string; rows: { t: string; pnl: number; label: string }[] }) {
  return (
    <div className="rounded-2xl border border-border bg-elevated/40 p-4">
      <h3 className="font-display font-semibold">{title}</h3>
      <div className="mt-4 h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows}>
            <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
            <XAxis dataKey="label" tick={{ fill: '#a1a1aa', fontSize: 11 }} minTickGap={16} />
            <YAxis tick={{ fill: '#a1a1aa', fontSize: 11 }} width={56} />
            <Tooltip
              contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
              formatter={(v: number | undefined) => brl(Number(v ?? 0))}
            />
            <Bar dataKey="pnl" radius={[6, 6, 0, 0]}>
              {rows.map((row) => (
                <Cell key={row.t} fill={row.pnl >= 0 ? '#34d399' : '#fb7185'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function SideMark({ side }: { side: string }) {
  const buy = side === 'BUY'
  const sell = side === 'SELL'
  return (
    <span
      className={cn(
        'rounded-lg px-2 py-0.5 text-xs font-semibold uppercase tracking-wide',
        buy && 'bg-gain/15 text-gain',
        sell && 'bg-loss/15 text-loss',
        !buy && !sell && 'bg-card text-muted-foreground',
      )}
    >
      {side}
    </span>
  )
}

export function LivePage() {
  const [snap, setSnap] = useState<LiveSnap>(EMPTY)
  const [offline, setOffline] = useState(false)
  const [source, setSource] = useState<'paper' | 'mt5'>('paper')
  const [intervalSec, setIntervalSec] = useState(0.001)
  const [busy, setBusy] = useState(false)
  const [caseKey, setCaseKey] = useState<CaseKey>('last_candles')
  const [timeframe, setTimeframe] = useState<TfKey>('m5')
  const [bank, setBank] = useState(1000)
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [meta, setMeta] = useState<LiveMeta | null>(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const res = await fetch(`/api/live/meta?timeframe=${timeframe}`, { cache: 'no-store' })
        if (!res.ok) throw new Error('fail')
        const json = (await res.json()) as LiveMeta
        if (!alive) return
        setMeta(json)
        setStart((prev) => prev || json.default_start)
        setEnd((prev) => prev || json.default_end)
      } catch {
        if (alive) setOffline(true)
      }
    }
    void load()
    return () => {
      alive = false
    }
  }, [timeframe])

  useEffect(() => {
    let alive = true
    let hydrated = false
    const pull = async () => {
      try {
        const res = await fetch('/api/live', { cache: 'no-store' })
        if (!res.ok) throw new Error('fail')
        const json = (await res.json()) as LiveSnap
        if (!alive) return
        setSnap(json)
        setOffline(false)
        if (json.source === 'paper' || json.source === 'mt5') setSource(json.source)
        if (json.interval_sec) setIntervalSec(json.interval_sec)
        if (!hydrated) {
          hydrated = true
          if (json.case) setCaseKey(asCase(json.case))
          if (json.timeframe) setTimeframe(asTf(json.timeframe))
          if (json.initial_bank) setBank(Number(json.initial_bank))
          const from = (json.start || json.window_start || '').slice(0, 10)
          const to = (json.end || json.window_end || '').slice(0, 10)
          if (from) setStart(from)
          if (to) setEnd(to)
        }
      } catch {
        if (alive) setOffline(true)
      }
    }
    void pull()
    const id = window.setInterval(pull, 1000)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [])

  const post = async (path: string, body?: object) => {
    setBusy(true)
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
      })
      const json = (await res.json()) as LiveSnap & { detail?: string }
      if (!res.ok) throw new Error(typeof json.detail === 'string' ? json.detail : 'falhou')
      setSnap(json)
      setOffline(false)
    } catch (err) {
      setSnap((prev) => ({ ...prev, error: err instanceof Error ? err.message : 'falhou' }))
    } finally {
      setBusy(false)
    }
  }

  const equity = useMemo(
    () =>
      snap.equity.map((row) => ({
        ...row,
        label: row.t.replace('T', ' ').slice(5, 16),
      })),
    [snap.equity],
  )
  const periodLevels = (snap.periods?.levels?.length
    ? snap.periods.levels
    : clientLevels(start, end)) as PeriodLevel[]
  const periodCharts = useMemo(
    () =>
      periodLevels.map((level) => {
        const rows = snap.periods?.series?.[level] ?? (level === 'daily' ? snap.daily : [])
        return {
          level,
          rows: rows.map((row) => ({
            ...row,
            label: periodTick(level, row.t),
          })),
        }
      }),
    [periodLevels, snap.daily, snap.periods],
  )
  const candles = useMemo(
    () =>
      snap.candles.map((row) => ({
        ...row,
        label: row.t.replace('T', ' ').slice(11, 16),
      })),
    [snap.candles],
  )

  const banks = meta?.banks?.length ? meta.banks : FALLBACK_BANKS
  const rangeMin = meta?.min_date || RANGE_MIN
  const rangeMax = meta?.max_date || new Date().toISOString().slice(0, 10)
  const startMax = end ? minIso(end, rangeMax) : rangeMax
  const endMin = start ? maxIso(start, rangeMin) : rangeMin
  const endMax = start ? minIso(rangeMax, addMonths(start, 3)) : rangeMax
  const dateError =
    source === 'paper' && start && end
      ? end < start
        ? 'Data final deve ser maior ou igual à inicial'
        : windowTooLong(start, end)
          ? 'Janela limitada a 3 meses (treino trimestral)'
          : start < rangeMin
            ? `Data inicial mínima: ${dayLabel(rangeMin)}`
            : end > rangeMax
              ? `Data final máxima: ${dayLabel(rangeMax)}`
              : null
      : null
  const locked = busy || snap.running
  const windowText =
    source === 'paper' && start && end
      ? `Janela ${dayLabel(start)} → ${dayLabel(end)}`
      : 'MT5 ao vivo (sem janela de CSV)'

  const status = offline
    ? 'API offline — suba python -m trader serve'
    : snap.running
      ? 'Operando (paper, sem ordem no MT5)'
      : snap.done
        ? 'Fim dos candles da sessão'
        : 'Pausado'

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <header className="relative mx-auto flex max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
        <p className="font-display text-lg font-bold">Sinal WIN</p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <a href="/">Estudo</a>
          </Button>
        </div>
      </header>

      <section className="relative mx-auto max-w-6xl px-5 pb-4 sm:px-8">
        <p className="text-sm uppercase tracking-[0.2em] text-primary">Ao vivo · local</p>
        <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-display text-4xl font-bold">Sinais e ordens</h1>
            <p className="mt-2 max-w-xl text-muted-foreground">
              {status}. {CASE_LABEL[caseKey]} · {TF_LABEL[timeframe]}. Lote sobe 1 mini a cada{' '}
              {brl(snap.initial_bank || bank)}. {windowText}
              {snap.last_bar_time ? ` · candle ${clock(snap.last_bar_time)}` : ''}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              className={selectClass}
              value={caseKey}
              onChange={(e) => setCaseKey(asCase(e.target.value))}
              disabled={locked}
            >
              <option value="last_candles">Últimos candles</option>
              <option value="last_candle">Último candle</option>
            </select>
            <select
              className={selectClass}
              value={bank}
              onChange={(e) => setBank(Number(e.target.value))}
              disabled={locked}
            >
              {banks.map((value) => (
                <option key={value} value={value}>
                  R$ {value.toLocaleString('pt-BR')}
                </option>
              ))}
            </select>
            <select
              className={selectClass}
              value={timeframe}
              onChange={(e) => setTimeframe(asTf(e.target.value))}
              disabled={locked}
            >
              <option value="m5">5 min</option>
              <option value="m1">1 min</option>
            </select>
            <label className="flex flex-col gap-1 text-[11px] uppercase tracking-wide text-muted-foreground">
              De
              <input
                type="date"
                className={selectClass}
                value={start}
                min={rangeMin}
                max={startMax || undefined}
                disabled={locked || source === 'mt5'}
                onChange={(e) => {
                  const next = e.target.value
                  setStart(next)
                  if (next && end && end < next) setEnd(next)
                }}
              />
            </label>
            <label className="flex flex-col gap-1 text-[11px] uppercase tracking-wide text-muted-foreground">
              Até
              <input
                type="date"
                className={selectClass}
                value={end}
                min={endMin}
                max={endMax || undefined}
                disabled={locked || source === 'mt5'}
                onChange={(e) => setEnd(e.target.value)}
              />
            </label>
            <select
              className={selectClass}
              value={source}
              onChange={(e) => setSource(e.target.value as 'paper' | 'mt5')}
              disabled={snap.running}
            >
              <option value="paper">Paper (CSV)</option>
              <option value="mt5">MT5 (Windows)</option>
            </select>
            <select
              className={selectClass}
              value={intervalSec}
              onChange={(e) => setIntervalSec(Number(e.target.value))}
              disabled={snap.running}
            >
              <option value={0.001}>1 ms / candle</option>
              <option value={1}>1 s / candle</option>
              <option value={300}>5 min (relógio)</option>
            </select>
            <Button
              size="sm"
              disabled={busy || snap.running || Boolean(dateError)}
              onClick={() =>
                void post('/api/live/start', {
                  case: caseKey,
                  timeframe,
                  initial_bank: bank,
                  start: source === 'paper' ? start : undefined,
                  end: source === 'paper' ? end : undefined,
                  source,
                  interval_sec: intervalSec,
                })
              }
            >
              Iniciar
            </Button>
            <Button variant="outline" size="sm" disabled={busy || !snap.running} onClick={() => void post('/api/live/stop')}>
              Pausar
            </Button>
            <Button variant="ghost" size="sm" disabled={busy || snap.running} onClick={() => void post('/api/live/reset')}>
              Zerar
            </Button>
          </div>
        </div>
        {dateError ? <p className="mt-3 text-sm text-loss">{dateError}</p> : null}
        {snap.error ? <p className="mt-3 text-sm text-loss">{snap.error}</p> : null}
      </section>

      <section className="relative mx-auto grid max-w-6xl gap-3 px-5 py-4 sm:grid-cols-2 sm:px-8 lg:grid-cols-4">
        <Kpi label="Banca" value={brl(snap.bank)} hint={`${snap.contracts} mini agora · teto ${snap.max_contracts}`} />
        <Kpi
          label="P&L sessão"
          value={brl(snap.net_pnl)}
          positive={snap.net_pnl === 0 ? null : snap.net_pnl > 0}
          hint={`tombo ${brl(snap.max_drawdown)} (${pct(snap.max_drawdown_pct)})`}
        />
        <Kpi
          label="P&L hoje"
          value={brl(snap.today_pnl)}
          positive={snap.today_pnl === 0 ? null : snap.today_pnl > 0}
          hint="dia do candle atual"
        />
        <Kpi
          label="Acerto"
          value={pct(snap.win_rate)}
          positive={snap.n_trades === 0 ? null : snap.win_rate >= 50}
          hint={snap.n_trades ? `${snap.n_wins}/${snap.n_trades} trades` : 'ainda sem trades'}
        />
      </section>

      <section
        className={cn(
          'relative mx-auto grid max-w-6xl gap-3 px-5 pb-4 sm:px-8',
          periodLevels.length >= 4 ? 'sm:grid-cols-2 lg:grid-cols-4' : periodLevels.length === 3 ? 'sm:grid-cols-3' : 'sm:grid-cols-2',
        )}
      >
        {periodLevels.map((level) => (
          <PeriodAvgCard key={level} level={level} stats={snap.periods?.avg?.[level]} />
        ))}
      </section>

      <section className="relative mx-auto grid max-w-6xl gap-4 px-5 py-2 sm:px-8 lg:grid-cols-3">
        <div className="rounded-2xl border border-border bg-elevated/50 p-5 lg:col-span-1">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Sinal atual</p>
          <div className="mt-3 flex items-center gap-3">
            <SideMark side={snap.signal?.side ?? 'FLAT'} />
            <p className="font-display text-xl font-bold">{snap.signal?.reason || 'aguardando'}</p>
          </div>
          <dl className="mt-4 grid grid-cols-3 gap-2 text-sm">
            <div>
              <dt className="text-muted-foreground">Entrada</dt>
              <dd className="tabular-nums">{snap.signal ? snap.signal.entry.toFixed(0) : '—'}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Stop</dt>
              <dd className="tabular-nums text-loss">{snap.signal ? snap.signal.stop.toFixed(0) : '—'}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Alvo</dt>
              <dd className="tabular-nums text-gain">{snap.signal ? snap.signal.take.toFixed(0) : '—'}</dd>
            </div>
          </dl>
          <p className="mt-4 text-xs text-muted-foreground">
            {snap.n_trades} trades · {snap.n_wins} wins · acerto {pct(snap.win_rate)} · barra {snap.cursor}/
            {snap.n_bars}
          </p>
        </div>
        <div className="rounded-2xl border border-border bg-elevated/50 p-5 lg:col-span-2">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Ordem aberta</p>
          {snap.position ? (
            <div className="mt-3 flex flex-wrap items-center gap-4">
              <SideMark side={snap.position.side} />
              <p className="tabular-nums">
                {snap.position.contracts ?? 1} mini · {snap.position.entry.toFixed(0)} · stop{' '}
                {snap.position.stop.toFixed(0)} · alvo {snap.position.take.toFixed(0)}
              </p>
              <p className="text-sm text-muted-foreground">{clock(snap.position.time)}</p>
            </div>
          ) : (
            <p className="mt-3 text-muted-foreground">Nenhuma posição aberta.</p>
          )}
          <div className="mt-6 h-40">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={candles}>
                <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fill: '#a1a1aa', fontSize: 11 }} minTickGap={24} />
                <YAxis domain={['auto', 'auto']} tick={{ fill: '#a1a1aa', fontSize: 11 }} width={56} />
                <Tooltip
                  contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
                  labelStyle={{ color: '#f5f5f7' }}
                />
                <Line type="monotone" dataKey="close" stroke="#058ef2" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="relative mx-auto grid max-w-6xl gap-4 px-5 py-4 sm:px-8 lg:grid-cols-2">
        <div className="rounded-2xl border border-border bg-elevated/40 p-4">
          <h3 className="font-display font-semibold">Banca</h3>
          <div className="mt-4 h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equity}>
                <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fill: '#a1a1aa', fontSize: 11 }} minTickGap={28} />
                <YAxis tick={{ fill: '#a1a1aa', fontSize: 11 }} width={56} />
                <Tooltip
                  contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
                  formatter={(v: number | undefined) => brl(Number(v ?? 0))}
                />
                <Area type="monotone" dataKey="bank" stroke="#058ef2" fill="#058ef233" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        {periodCharts.map((chart) => (
          <PeriodChart
            key={chart.level}
            title={`P&L ${PERIOD_LABEL[chart.level].toLowerCase()}`}
            rows={chart.rows}
          />
        ))}
      </section>

      <section className="relative mx-auto max-w-6xl px-5 py-6 sm:px-8">
        <h3 className="font-display font-semibold">Ordens</h3>
        <div className="mt-4 overflow-x-auto rounded-2xl border border-border">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead className="bg-elevated/80 text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-3">Entrada</th>
                <th className="px-4 py-3">Saída</th>
                <th className="px-4 py-3">Lado</th>
                <th className="px-4 py-3">Minis</th>
                <th className="px-4 py-3">Pts</th>
                <th className="px-4 py-3">P&L</th>
                <th className="px-4 py-3">Motivo</th>
              </tr>
            </thead>
            <tbody>
              {snap.trades.length === 0 ? (
                <tr>
                  <td className="px-4 py-6 text-muted-foreground" colSpan={7}>
                    Nenhuma ordem ainda. Inicie a sessão paper.
                  </td>
                </tr>
              ) : (
                snap.trades.map((t) => (
                  <tr key={`${t.entry_time}-${t.exit_time}-${t.pnl}`} className="border-t border-border/70">
                    <td className="px-4 py-2 tabular-nums">{clock(t.entry_time)}</td>
                    <td className="px-4 py-2 tabular-nums">{clock(t.exit_time)}</td>
                    <td className="px-4 py-2">
                      <SideMark side={t.side} />
                    </td>
                    <td className="px-4 py-2 tabular-nums">{t.contracts ?? 1}</td>
                    <td className="px-4 py-2 tabular-nums">{t.points.toFixed(0)}</td>
                    <td className={cn('px-4 py-2 tabular-nums', t.pnl >= 0 ? 'text-gain' : 'text-loss')}>
                      {brl(t.pnl)}
                    </td>
                    <td className="px-4 py-2 text-muted-foreground">{t.reason}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
