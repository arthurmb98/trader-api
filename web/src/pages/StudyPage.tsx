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
import { Activity, AlertTriangle, CheckCircle2, Cpu, LineChart, Shield } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { brl, cn, num, pct } from '@/lib/utils'
import type { BankKey, PeriodRow, StudyFile, Winner } from '@/lib/types'

function stopGainLabel(winner: Winner) {
  const risk = winner.params.risk
  const mode = String(risk.mode)
  const trailing = Boolean(risk.trailing_enabled)
  if (mode === 'atr') {
    return {
      stop: `ATR × ${risk.atr_stop_mult}`,
      gain: `ATR × ${risk.atr_gain_mult}`,
      trailing,
    }
  }
  return {
    stop: `${risk.stop_points} pts`,
    gain: `${risk.gain_points} pts`,
    trailing,
  }
}

const PARAM_LABELS: Record<string, string> = {
  initial_bank: 'Banca inicial',
  contracts: 'Contratos',
  point_value: 'Valor do ponto',
  contract_cost: 'Custo por operação',
  mode: 'Tipo de stop/gain',
  stop_points: 'Stop (pontos)',
  gain_points: 'Alvo (pontos)',
  rr_ratio: 'Risco/retorno',
  atr_period: 'ATR (períodos)',
  atr_stop_mult: 'Stop × ATR',
  atr_gain_mult: 'Alvo × ATR',
  trailing_enabled: 'Stop móvel',
  trailing_trigger_points: 'Ativa trailing em',
  trailing_distance_points: 'Distância do trailing',
  daily_loss_points: 'Stop diário (pontos)',
  max_trades_per_day: 'Máx. operações/dia',
  session_start: 'Início da sessão',
  session_end: 'Fim da sessão',
  skip_lunch: 'Evitar almoço',
  lunch_start: 'Almoço de',
  lunch_end: 'Almoço até',
  gold_hours_only: 'Só horário-ouro',
  min_predicted_range: 'Range mínimo previsto',
  max_gap_points: 'Gap máximo',
  min_predicted_body: 'Corpo mínimo previsto',
  direction: 'Direção',
  entry_mode: 'Tipo de entrada',
  entry_offset_points: 'Offset da ordem',
  timeframe: 'Tempo gráfico',
  train_csv: 'Arquivo de treino',
  test_csv: 'Arquivo de teste',
  symbol: 'Símbolo',
  tick_size: 'Tick',
}

function formatParam(key: string, value: unknown) {
  if (typeof value === 'boolean') return value ? 'sim' : 'não'
  if (value === null || value === undefined) return 'sem limite'
  if (key.includes('bank') || key === 'point_value' || key === 'contract_cost') {
    return brl(Number(value))
  }
  if (key === 'direction') return value === 'follow' ? 'seguir a previsão' : 'contra a previsão (fade)'
  if (key === 'entry_mode') return value === 'market_open' ? 'a mercado na abertura' : 'limitada dentro do range'
  if (key === 'mode') {
    const map: Record<string, string> = {
      fixed: 'stop e alvo fixos',
      rr: 'alvo em múltiplo do stop',
      atr: 'stop/alvo pelo ATR',
    }
    return map[String(value)] ?? String(value)
  }
  return String(value)
}

function Kpi({ label, value, hint, positive }: { label: string; value: string; hint?: string; positive?: boolean | null }) {
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
  compound,
}: {
  winner: Winner
  title: string
  active: boolean
  onClick: () => void
  compound: boolean
}) {
  const m = compound && winner.compound ? winner.compound.metrics : winner.metrics
  const sg = stopGainLabel(winner)
  const y26 = (compound ? winner.compound?.by_year : winner.by_year)?.['2026']
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'rounded-2xl border p-5 text-left transition-colors',
        active ? 'border-primary bg-card' : 'border-border bg-elevated/50 hover:bg-card/80',
      )}
    >
      <p className="text-sm text-muted-foreground">{title}</p>
      <p className="mt-1 font-display text-xl font-semibold">{winner.params.name}</p>
      <p className={cn('mt-3 text-3xl font-bold', m.net_pnl >= 0 ? 'text-gain' : 'text-loss')}>{brl(m.net_pnl)}</p>
      <p className="mt-2 text-sm text-foreground/90">
        Stop {sg.stop} · Alvo {sg.gain}
        {sg.trailing ? ' · trailing' : ''}
      </p>
      <p className="mt-1 text-sm text-muted-foreground">
        {m.n_trades} operações · acerto {pct(m.win_rate)} · fator {num(m.profit_factor, 2)}
      </p>
      {y26 ? (
        <p className={cn('mt-2 text-xs', y26.net_pnl >= 0 ? 'text-gain' : 'text-loss')}>
          2026: {brl(y26.net_pnl)}
        </p>
      ) : null}
    </button>
  )
}

function visibleParams(title: string, obj: Record<string, unknown>, mode: string) {
  return Object.entries(obj).filter(([k]) => {
    if (title === 'Risco' && mode === 'atr' && (k === 'stop_points' || k === 'gain_points' || k === 'rr_ratio')) return false
    if (title === 'Risco' && mode === 'fixed' && (k.startsWith('atr_') || k === 'rr_ratio')) return false
    if (title === 'Risco' && mode === 'rr' && (k.startsWith('atr_') || k === 'gain_points')) return false
    if (title === 'Risco' && k.startsWith('trailing') && obj.trailing_enabled === false && k !== 'trailing_enabled') return false
    if (title === 'Execução' && obj.entry_mode !== 'limit_inside' && k === 'entry_offset_points') return false
    return true
  })
}

function ParamGrid({ winner }: { winner: Winner }) {
  const mode = String(winner.params.risk.mode)
  const blocks = [
    ['Conta', winner.params.account],
    ['Risco', winner.params.risk],
    ['Filtros', winner.params.filters],
    ['Execução', winner.params.execution],
    ['Dados', winner.params.data],
  ] as const
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {blocks.map(([title, obj]) => (
        <div key={title} className="rounded-2xl border border-border bg-elevated/50 p-4">
          <h4 className="font-display text-sm font-semibold text-primary">{title}</h4>
          <dl className="mt-3 space-y-2">
            {visibleParams(title, obj as Record<string, unknown>, mode).map(([k, v]) => (
              <div key={k} className="flex items-start justify-between gap-4 text-sm">
                <dt className="text-muted-foreground">{PARAM_LABELS[k] ?? k}</dt>
                <dd className="text-right font-medium">{formatParam(k, v)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  )
}

export function StudyPage() {
  const [data, setData] = useState<StudyFile | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [bank, setBank] = useState<BankKey>('1000')
  const [pick, setPick] = useState<{ tf: 'm1' | 'm5'; i: number }>({ tf: 'm1', i: 0 })
  const [compound, setCompound] = useState(false)

  useEffect(() => {
    const load = async () => {
      const urls = ['/studies.json', '/api/studies']
      for (const url of urls) {
        try {
          const res = await fetch(url)
          if (!res.ok) continue
          const json = (await res.json()) as StudyFile
          setData(json)
          const first = String(json.banks?.[1] ?? json.banks?.[0] ?? 1000) as BankKey
          if (json.winners[first]) setBank(first)
          return
        } catch {
          /* try next */
        }
      }
      setError('Estudo ainda não disponível. Rode python -m trader study.')
    }
    void load()
  }, [])

  const winner = data?.winners[bank]?.[pick.tf]?.[pick.i]
  const side = compound && winner?.compound ? winner.compound : winner
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
  if (!data || !winner || !side) {
    return (
      <main className="grid min-h-dvh place-items-center">
        <p className="animate-fade-in text-muted-foreground">Carregando o estudo…</p>
      </main>
    )
  }

  const m = side.metrics
  const leak = data.timeframes.m1.leakage
  const tfBlock = data.timeframes[pick.tf]
  const bankWinners = data.winners[bank]
  const nConfigsBank = (data.studies[bank]?.m1.n_configs ?? 0) + (data.studies[bank]?.m5.n_configs ?? 0)
  const years = side.by_year ?? {}
  const periods = side.by_period
  const sg = stopGainLabel(winner)
  const hasCompound = Boolean(winner.compound)

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <header className="relative mx-auto flex max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
        <p className="font-display text-lg font-bold">Sinal WIN</p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => document.getElementById('setups')?.scrollIntoView()}>
            Ver setups
          </Button>
          <Button size="sm" onClick={() => document.getElementById('mt5')?.scrollIntoView()}>
            Caminho até o MT5
          </Button>
        </div>
      </header>

      <section className="relative mx-auto max-w-6xl px-5 pb-16 pt-10 sm:px-8">
        <p className="animate-fade-up text-sm uppercase tracking-[0.2em] text-primary">Estudo para sócios</p>
        <h1 className="mt-4 max-w-3xl animate-fade-up font-display text-4xl font-bold leading-tight sm:text-6xl [animation-delay:80ms]">
          Um robô que lê o último candle e sugere compra ou venda.
        </h1>
        <p className="mt-5 max-w-2xl animate-fade-up text-lg text-muted-foreground [animation-delay:140ms]">
          Mini índice WIN$ contínuo. Ranking oficial com 1 contrato. Treino de ago/2021 a dez/2024. Teste de jan/2025 a ago/2026.
          O modelo nunca vê os candles do teste. Três bancas — duas melhores de 1 min e duas de 5 min em cada uma. Há uma simulação extra que dobra contratos quando a banca dobra.
        </p>
        <div className="mt-8 grid gap-3 sm:grid-cols-3">
          <div className="rounded-2xl border border-border bg-elevated/60 p-4">
            <Cpu className="size-5 text-primary" />
            <p className="mt-3 font-display font-semibold">Parede em 2025</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Treino até 31/12/2024, teste a partir de 02/01/2025. {leak.n_removed} candles repetidos
              removidos de {leak.n_test_original.toLocaleString('pt-BR')}.
            </p>
          </div>
          <div className="rounded-2xl border border-border bg-elevated/60 p-4">
            <LineChart className="size-5 text-violet" />
            <p className="mt-3 font-display font-semibold">{data.n_configs_total.toLocaleString('pt-BR')} combinações</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Stop fixo, risco/retorno, ATR, trailing, horário-ouro e fade — ranking só no teste, com piso de 40 operações e teto de drawdown.
            </p>
          </div>
          <div className="rounded-2xl border border-border bg-elevated/60 p-4">
            <Shield className="size-5 text-pink" />
            <p className="mt-3 font-display font-semibold">WIN$ contínuo</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Não é um vencimento só: há saltos de rolagem. O grid filtra gap grande. MT5 desligado até o paper.
            </p>
          </div>
        </div>
      </section>

      <section className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <h2 className="font-display text-3xl font-bold">Como ele decide</h2>
        <ol className="mt-6 grid gap-4 sm:grid-cols-2">
          {data.how_it_works.map((step, i) => (
            <li key={step} className="rounded-2xl border border-border bg-elevated/50 p-5">
              <span className="font-display text-sm text-primary">0{i + 1}</span>
              <p className="mt-2 text-foreground/90">{step}</p>
            </li>
          ))}
        </ol>
        <p className="mt-6 rounded-2xl border border-border bg-card/60 p-4 text-sm text-muted-foreground">
          Ponto do mini índice = {brl(data.instrument.point_value)}. Tick = {data.instrument.tick} pontos. Sempre 1
          contrato. Banca simulada selecionada: {brl(m.initial_bank)}.
        </p>
      </section>

      <section id="setups" className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <h2 className="font-display text-3xl font-bold">As melhores estratégias por banca</h2>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          {nConfigsBank.toLocaleString('pt-BR')} combinações nesta banca. Clique em um card para ver a curva, 2025 vs 2026 e cada parâmetro.
        </p>
        <div className="mt-5 flex flex-wrap gap-2">
          {(data.banks.length ? data.banks : [500, 1000, 5000]).map((value) => {
            const key = String(value) as BankKey
            return (
              <Button
                key={key}
                variant={bank === key ? 'default' : 'outline'}
                size="sm"
                onClick={() => {
                  setBank(key)
                  setPick({ tf: 'm1', i: 0 })
                }}
              >
                {brl(value)}
              </Button>
            )
          })}
        </div>
        </div>
        {hasCompound ? (
          <div className="mt-4 flex flex-wrap gap-2">
            <Button variant={!compound ? 'default' : 'outline'} size="sm" onClick={() => setCompound(false)}>
              1 contrato (ranking)
            </Button>
            <Button variant={compound ? 'default' : 'outline'} size="sm" onClick={() => setCompound(true)}>
              Contratos ×2 a cada 2× da banca
            </Button>
          </div>
        ) : null}
        {compound ? (
          <p className="mt-3 text-sm text-muted-foreground">
            Ilustração de alavancagem: 1 contrato até 2× a banca inicial, 2 até 4×, 4 até 8×, teto 16. Não é o critério do ranking.
            {m.max_contracts ? ` Chegou a ${m.max_contracts} contratos.` : ''}
          </p>
        ) : null}
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {bankWinners.m1.map((w, i) => (
            <WinnerCard
              key={w.name}
              winner={w}
              title={`1 minuto · ${i === 0 ? 'melhor' : 'segundo melhor'}`}
              active={pick.tf === 'm1' && pick.i === i}
              onClick={() => setPick({ tf: 'm1', i })}
              compound={compound}
            />
          ))}
          {bankWinners.m5.map((w, i) => (
            <WinnerCard
              key={w.name}
              winner={w}
              title={`5 minutos · ${i === 0 ? 'melhor' : 'segundo melhor'}`}
              active={pick.tf === 'm5' && pick.i === i}
              onClick={() => setPick({ tf: 'm5', i })}
              compound={compound}
            />
          ))}
        </div>

        <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Kpi label="Lucro no teste" value={brl(m.net_pnl)} positive={m.net_pnl >= 0} />
          <Kpi label="Stop" value={sg.stop} />
          <Kpi label="Alvo" value={sg.gain} hint={sg.trailing ? 'com trailing' : undefined} />
          <Kpi label="Taxa de acerto" value={pct(m.win_rate)} hint={`${m.n_wins} gains · ${m.n_losses} stops`} />
          <Kpi label="Média / mediana do ganho" value={`${brl(m.avg_win)} · ${brl(m.median_win)}`} positive />
          <Kpi label="Média / mediana da perda" value={`${brl(m.avg_loss)} · ${brl(m.median_loss)}`} positive={false} />
          <Kpi label="Fator de lucro" value={num(m.profit_factor, 2)} hint="Acima de 1 = ganhos cobrem perdas" />
          <Kpi label="Maior tombo" value={brl(m.max_drawdown)} hint={pct(m.max_drawdown_pct)} />
          <Kpi label="Banca final" value={brl(m.final_bank)} />
          <Kpi
            label="Acerto de direção do ML"
            value={pct(tfBlock.model_test.test_direction_hit * 100)}
            hint="Só no arquivo de teste"
          />
        </div>

        <div className="mt-8 grid gap-3 sm:grid-cols-2">
          {Object.entries(years).map(([year, slice]) => (
            <div key={year} className="rounded-2xl border border-border bg-elevated/50 p-4">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Robustez · {year}</p>
              <p className={cn('mt-2 font-display text-2xl font-bold', slice.net_pnl >= 0 ? 'text-gain' : 'text-loss')}>
                {brl(slice.net_pnl)}
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                {slice.n_trades} operações · acerto {pct(slice.win_rate)} · tombo {pct(slice.max_drawdown_pct)}
              </p>
            </div>
          ))}
        </div>

        {periods ? (
          <div className="mt-8 space-y-6">
            <h3 className="font-display text-2xl font-bold">Ganho e perda por período</h3>
            <div className="grid gap-3 sm:grid-cols-3">
              <Kpi
                label="Melhor / pior dia"
                value={`${brl(periods.summary.day.best?.pnl ?? 0)} · ${brl(periods.summary.day.worst?.pnl ?? 0)}`}
                hint={`${pct(periods.summary.day.positive_pct)} dos dias no azul · média ${brl(periods.summary.day.avg)}`}
              />
              <Kpi
                label="Melhor / pior semana"
                value={`${brl(periods.summary.week.best?.pnl ?? 0)} · ${brl(periods.summary.week.worst?.pnl ?? 0)}`}
                hint={`média ${brl(periods.summary.week.avg)}`}
              />
              <Kpi
                label="Melhor / pior mês"
                value={`${brl(periods.summary.month.best?.pnl ?? 0)} · ${brl(periods.summary.month.worst?.pnl ?? 0)}`}
                hint={`média ${brl(periods.summary.month.avg)}`}
              />
            </div>
            <PeriodBars title="Diário" rows={periods.daily} />
            <PeriodBars title="Semanal" rows={periods.weekly} />
            <PeriodBars title="Mensal" rows={periods.monthly} />
          </div>
        ) : null}

        <div className="mt-8 grid gap-6 lg:grid-cols-5">
          <div className="rounded-2xl border border-border bg-elevated/40 p-4 lg:col-span-3">
            <h3 className="font-display font-semibold">Curva da banca</h3>
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
            <h3 className="font-display font-semibold">Gains vs stops</h3>
            <div className="mt-4 h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[{ name: 'Gains', v: m.n_wins }, { name: 'Stops', v: m.n_losses }]}>
                  <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fill: '#a1a1aa' }} />
                  <YAxis tick={{ fill: '#a1a1aa' }} allowDecimals={false} />
                  <Tooltip contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }} />
                  <Bar dataKey="v" radius={[8, 8, 0, 0]}>
                    <Cell fill="#34d399" />
                    <Cell fill="#fb7185" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-border bg-elevated/40 p-4">
          <h3 className="font-display font-semibold">Resultado por horário</h3>
          <div className="mt-4 h-64">
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

        <h3 className="mt-10 font-display text-2xl font-bold">Todos os parâmetros deste setup</h3>
        <p className="mt-2 text-sm text-muted-foreground">O que estiver neste quadro é exatamente o que o robô usou no teste.</p>
        <div className="mt-4">
          <ParamGrid winner={winner} />
        </div>
      </section>

      <section className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <h2 className="font-display text-3xl font-bold">O que deu certo, o que não deu, o que melhorar</h2>
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <div className="rounded-2xl border border-border bg-elevated/50 p-5">
            <CheckCircle2 className="size-5 text-gain" />
            <h3 className="mt-3 font-display font-semibold">Funcionou</h3>
            <ul className="mt-3 space-y-2 text-sm text-foreground/90">
              {data.insights.worked.map((item) => (
                <li key={item}>· {item}</li>
              ))}
            </ul>
          </div>
          <div className="rounded-2xl border border-border bg-elevated/50 p-5">
            <AlertTriangle className="size-5 text-pink" />
            <h3 className="mt-3 font-display font-semibold">Não funcionou</h3>
            <ul className="mt-3 space-y-2 text-sm text-foreground/90">
              {data.insights.failed.map((item) => (
                <li key={item}>· {item}</li>
              ))}
            </ul>
          </div>
          <div className="rounded-2xl border border-border bg-elevated/50 p-5">
            <Activity className="size-5 text-primary" />
            <h3 className="mt-3 font-display font-semibold">Próximo passo</h3>
            <ul className="mt-3 space-y-2 text-sm text-foreground/90">
              {data.insights.improve.map((item) => (
                <li key={item}>· {item}</li>
              ))}
            </ul>
          </div>
        </div>
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
          <img
            src="/brand/logo-branco.png"
            alt="Koletivo Hub"
            className="h-10 w-auto object-contain"
          />
          <span className="font-display text-sm font-semibold">Desenvolvido por Koletivo Hub</span>
        </a>
        <p className="mt-3 text-xs text-muted-foreground/70">
          © {new Date().getFullYear()}. Todos os direitos reservados a Koletivo Hub.
        </p>
      </footer>
    </div>
  )
}
