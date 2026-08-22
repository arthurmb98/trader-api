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

type LiveSnap = {
  running: boolean
  done: boolean
  error: string | null
  config: string
  source: string
  interval_sec: number
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
}

function clock(value: string | null | undefined) {
  if (!value) return '—'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return value.replace('T', ' ').slice(0, 19)
  return d.toLocaleString('pt-BR')
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

  useEffect(() => {
    let alive = true
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
  const daily = useMemo(
    () =>
      snap.daily.map((row) => ({
        ...row,
        label: row.t.slice(5),
      })),
    [snap.daily],
  )
  const candles = useMemo(
    () =>
      snap.candles.map((row) => ({
        ...row,
        label: row.t.replace('T', ' ').slice(11, 16),
      })),
    [snap.candles],
  )

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
              {status}. Lote sobe 1 mini a cada R$ {snap.initial_bank.toFixed(0)} de banca. Semana = últimos 5
              pregões do CSV
              {snap.last_bar_time ? ` · candle ${clock(snap.last_bar_time)}` : ''}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              className="h-9 rounded-lg border border-border bg-elevated/60 px-3 text-sm"
              value={source}
              onChange={(e) => setSource(e.target.value as 'paper' | 'mt5')}
              disabled={snap.running}
            >
              <option value="paper">Paper (CSV)</option>
              <option value="mt5">MT5 (Windows)</option>
            </select>
            <select
              className="h-9 rounded-lg border border-border bg-elevated/60 px-3 text-sm"
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
              disabled={busy || snap.running}
              onClick={() =>
                void post('/api/live/start', {
                  config: 'best_candles_m5_1000_a',
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
          label="Média diária"
          value={brl(snap.avg_daily)}
          positive={snap.avg_daily === 0 ? null : snap.avg_daily > 0}
          hint={snap.n_days ? `${snap.n_days} dia(s) com trade` : 'ainda sem trades'}
        />
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
        <div className="rounded-2xl border border-border bg-elevated/40 p-4">
          <h3 className="font-display font-semibold">P&L por dia</h3>
          <div className="mt-4 h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={daily}>
                <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fill: '#a1a1aa', fontSize: 11 }} />
                <YAxis tick={{ fill: '#a1a1aa', fontSize: 11 }} width={56} />
                <Tooltip
                  contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
                  formatter={(v: number | undefined) => brl(Number(v ?? 0))}
                />
                <Bar dataKey="pnl" radius={[6, 6, 0, 0]}>
                  {daily.map((row) => (
                    <Cell key={row.t} fill={row.pnl >= 0 ? '#34d399' : '#fb7185'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
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
