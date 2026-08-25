import { useMemo } from 'react'
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
import {
  PERIOD_LABEL,
  asLot,
  clock,
  type LiveSnap,
  type PeriodAvg,
  type PeriodLevel,
} from '@/lib/liveTypes'
import { brl, cn, pct } from '@/lib/utils'

const PERIOD_UNIT: Record<PeriodLevel, string> = {
  daily: 'dia',
  weekly: 'semana',
  monthly: 'mês',
  quarterly: 'trimestre',
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

export function SideMark({ side }: { side: string }) {
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

export function SessionNav() {
  return (
    <header className="relative mx-auto flex max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
      <p className="font-display text-lg font-bold">Sinal WIN</p>
      <div className="flex gap-2">
        <Button variant="outline" size="sm" asChild>
          <a href="/">Estudo</a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a href="/replay">Replay</a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a href="/ao-vivo">Ao vivo</a>
        </Button>
      </div>
    </header>
  )
}

export function SessionDashboard({ snap, emptyHint }: { snap: LiveSnap; emptyHint: string }) {
  const equity = useMemo(
    () => snap.equity.map((row) => ({ ...row, label: row.t.replace('T', ' ').slice(5, 16) })),
    [snap.equity],
  )
  const periodLevels = (snap.periods?.levels?.length ? snap.periods.levels : ['daily']) as PeriodLevel[]
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
    () => snap.candles.map((row) => ({ ...row, label: row.t.replace('T', ' ').slice(11, 16) })),
    [snap.candles],
  )

  return (
    <>
      <section className="relative mx-auto grid max-w-6xl gap-3 px-5 py-4 sm:grid-cols-2 sm:px-8 lg:grid-cols-4">
        <Kpi
          label="Banca"
          value={brl(snap.bank)}
          hint={
            asLot(snap.lot) === 'scaled'
              ? `${brl(snap.initial_bank)} inicial · ${snap.contracts} mini agora · teto ${snap.max_contracts}`
              : `${brl(snap.initial_bank)} inicial · sempre 1 mini`
          }
        />
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
          periodLevels.length >= 4
            ? 'sm:grid-cols-2 lg:grid-cols-4'
            : periodLevels.length === 3
              ? 'sm:grid-cols-3'
              : 'sm:grid-cols-2',
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
            {snap.n_trades} trades · {snap.n_wins} wins · acerto {pct(snap.win_rate)} · barra {snap.cursor}/{snap.n_bars}
          </p>
        </div>
        <div className="rounded-2xl border border-border bg-elevated/50 p-5 lg:col-span-2">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Ordem aberta</p>
          {snap.position ? (
            <div className="mt-3 space-y-3">
              <div className="flex flex-wrap items-center gap-4">
                <SideMark side={snap.position.side} />
                <p className="tabular-nums">
                  {snap.position.contracts ?? 1} mini · {snap.position.entry.toFixed(0)} · stop {snap.position.stop.toFixed(0)}{' '}
                  · alvo {snap.position.take.toFixed(0)}
                </p>
                <p className="text-sm text-muted-foreground">{clock(snap.position.time)}</p>
                {snap.position.ticket ? (
                  <p className="text-xs text-muted-foreground">ticket {snap.position.ticket}</p>
                ) : (
                  <p className="text-xs text-muted-foreground">paper · sem envio</p>
                )}
              </div>
              <dl className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                <div>
                  <dt className="text-muted-foreground">Preço agora</dt>
                  <dd className="tabular-nums">{snap.position.mark != null ? snap.position.mark.toFixed(0) : '—'}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">P&L aberto</dt>
                  <dd
                    className={cn(
                      'tabular-nums font-semibold',
                      (snap.position.pnl ?? 0) > 0 && 'text-gain',
                      (snap.position.pnl ?? 0) < 0 && 'text-loss',
                    )}
                  >
                    {snap.position.pnl != null ? brl(snap.position.pnl) : '—'}
                    {snap.position.points != null ? ` · ${snap.position.points.toFixed(0)} pts` : ''}
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Até o stop</dt>
                  <dd className="tabular-nums text-loss">
                    {snap.position.to_stop != null ? `${snap.position.to_stop.toFixed(0)} pts` : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Até o gain</dt>
                  <dd className="tabular-nums text-gain">
                    {snap.position.to_take != null ? `${snap.position.to_take.toFixed(0)} pts` : '—'}
                  </dd>
                </div>
              </dl>
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
        <h3 className="font-display font-semibold">Sinais</h3>
        <div className="mt-4 overflow-x-auto rounded-2xl border border-border">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="bg-elevated/80 text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-3">Hora</th>
                <th className="px-4 py-3">Lado</th>
                <th className="px-4 py-3">Motivo</th>
                <th className="px-4 py-3">Entrada</th>
                <th className="px-4 py-3">Stop</th>
                <th className="px-4 py-3">Alvo</th>
              </tr>
            </thead>
            <tbody>
              {snap.signals.length === 0 ? (
                <tr>
                  <td className="px-4 py-6 text-muted-foreground" colSpan={6}>
                    Ainda sem sinal neste pregão. O motor reavalia a cada segundo no M5 do MT5.
                  </td>
                </tr>
              ) : (
                snap.signals.slice(0, 12).map((s) => (
                  <tr key={`${s.t}-${s.side}-${s.reason}`} className="border-t border-border/70">
                    <td className="px-4 py-2 tabular-nums">{clock(s.t)}</td>
                    <td className="px-4 py-2">
                      <SideMark side={s.side} />
                    </td>
                    <td className="px-4 py-2 text-muted-foreground">{s.reason}</td>
                    <td className="px-4 py-2 tabular-nums">{s.entry ? s.entry.toFixed(0) : '—'}</td>
                    <td className="px-4 py-2 tabular-nums text-loss">{s.stop ? s.stop.toFixed(0) : '—'}</td>
                    <td className="px-4 py-2 tabular-nums text-gain">{s.take ? s.take.toFixed(0) : '—'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
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
                    {emptyHint}
                  </td>
                </tr>
              ) : (
                snap.trades.map((t) => (
                  <tr key={`${t.entry_time}-${t.exit_time}-${t.pnl}-${t.result}`} className="border-t border-border/70">
                    <td className="px-4 py-2 tabular-nums">{clock(t.entry_time)}</td>
                    <td className="px-4 py-2 tabular-nums">{t.result === 'open' ? 'aberta' : clock(t.exit_time)}</td>
                    <td className="px-4 py-2">
                      <SideMark side={t.side} />
                    </td>
                    <td className="px-4 py-2 tabular-nums">{t.contracts ?? 1}</td>
                    <td className="px-4 py-2 tabular-nums">{t.points.toFixed(0)}</td>
                    <td className={cn('px-4 py-2 tabular-nums', t.pnl >= 0 ? 'text-gain' : 'text-loss')}>{brl(t.pnl)}</td>
                    <td className="px-4 py-2 text-muted-foreground">
                      {t.result === 'open' ? 'em aberto · ' : ''}
                      {t.reason}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </>
  )
}
