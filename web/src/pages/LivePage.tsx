import { useEffect, useState, type ReactNode } from 'react'
import { SessionDashboard, SessionNav } from '@/components/SessionDashboard'
import { Button } from '@/components/ui/button'
import {
  CASE_LABEL,
  EMPTY_SNAP,
  LOT_LABEL,
  TF_LABEL,
  asCase,
  asLot,
  asTf,
  clock,
  dayLabel,
  type CaseKey,
  type LiveMeta,
  type LiveSnap,
  type LotKey,
  type TfKey,
} from '@/lib/liveTypes'
import { readJson } from '@/lib/utils'

const RANGE_MIN = '2025-01-01'
const FALLBACK_BANKS = [500, 1000, 2000, 3000, 5000, 10000, 15000]
const selectClass = 'h-9 rounded-lg border border-border bg-elevated/60 px-3 text-sm'

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

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex min-w-0 flex-col justify-end gap-1 text-[11px] uppercase tracking-wide text-muted-foreground">
      {label}
      {children}
    </label>
  )
}

export function LivePage() {
  const [snap, setSnap] = useState<LiveSnap>(EMPTY_SNAP)
  const [offline, setOffline] = useState(false)
  const [source, setSource] = useState<'paper' | 'mt5'>('paper')
  const [intervalSec, setIntervalSec] = useState(0.001)
  const [busy, setBusy] = useState(false)
  const [caseKey, setCaseKey] = useState<CaseKey>('last_candles')
  const [timeframe, setTimeframe] = useState<TfKey>('m5')
  const [bank, setBank] = useState(1000)
  const [lot, setLot] = useState<LotKey>('fixed')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [meta, setMeta] = useState<LiveMeta | null>(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const res = await fetch(`/api/live/meta?timeframe=${timeframe}`, { cache: 'no-store' })
        if (!res.ok) throw new Error('fail')
        const json = await readJson<LiveMeta>(res)
        if (!alive) return
        setMeta(json)
        setOffline(false)
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
        if (!res.ok) return
        const json = await readJson<LiveSnap>(res)
        if (!alive) return
        setSnap((prev) => {
          if ((prev.done || prev.n_trades > 0) && !json.running && !(json.n_trades > 0)) {
            return prev
          }
          return json
        })
        setOffline(false)
        if (json.source === 'paper' || json.source === 'mt5') setSource(json.source)
        if (json.interval_sec) setIntervalSec(json.interval_sec)
        if (!hydrated) {
          hydrated = true
          if (json.case) setCaseKey(asCase(json.case))
          if (json.timeframe) setTimeframe(asTf(json.timeframe))
          if (json.initial_bank) setBank(Number(json.initial_bank))
          if (json.lot) setLot(asLot(json.lot))
          const from = (json.start || json.window_start || '').slice(0, 10)
          const to = (json.end || json.window_end || '').slice(0, 10)
          if (from) setStart(from)
          if (to) setEnd(to)
        }
      } catch {
        /* paper batch does not need a persistent GET */
      }
    }
    void pull()
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    if (!snap.running) return
    const id = window.setInterval(async () => {
      try {
        const res = await fetch('/api/live', { cache: 'no-store' })
        if (!res.ok) return
        const json = await readJson<LiveSnap>(res)
        setSnap(json)
        setOffline(false)
      } catch {
        setOffline(true)
      }
    }, 1000)
    return () => window.clearInterval(id)
  }, [snap.running])

  const post = async (path: string, body?: object) => {
    setBusy(true)
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
      })
      const json = await readJson<LiveSnap & { detail?: string }>(res)
      if (!res.ok) throw new Error(typeof json.detail === 'string' ? json.detail : 'falhou')
      setSnap(json)
      setOffline(false)
    } catch (err) {
      setSnap((prev) => ({ ...prev, error: err instanceof Error ? err.message : 'falhou' }))
    } finally {
      setBusy(false)
    }
  }

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

  const status =
    offline && !snap.done
      ? 'API offline — no Vercel o Iniciar chama a simulação paper; localmente use python -m trader serve'
      : busy
        ? 'Simulando a janela…'
        : snap.running
          ? 'Replay (paper, sem ordem no MT5)'
          : snap.done
            ? 'Simulação da janela concluída'
            : 'Pausado'

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <SessionNav />

      <section className="relative mx-auto max-w-6xl px-5 pb-4 sm:px-8">
        <p className="text-sm uppercase tracking-[0.2em] text-primary">Replay · local</p>
        <h1 className="mt-3 font-display text-4xl font-bold">Sinais e ordens</h1>
        <p className="mt-2 max-w-3xl text-muted-foreground">
          {status}. {CASE_LABEL[caseKey]} · {TF_LABEL[timeframe]} · {LOT_LABEL[lot]}. {windowText}
          {snap.last_bar_time ? ` · candle ${clock(snap.last_bar_time)}` : ''}
        </p>
        <div className="mt-4 flex flex-wrap items-end gap-2">
          <Field label="Caso">
            <select
              className={selectClass}
              value={caseKey}
              onChange={(e) => setCaseKey(asCase(e.target.value))}
              disabled={locked}
            >
              <option value="last_candles">Últimos candles</option>
              <option value="last_candle">Último candle</option>
            </select>
          </Field>
          <Field label="Banca">
            <select className={selectClass} value={bank} onChange={(e) => setBank(Number(e.target.value))} disabled={locked}>
              {banks.map((value) => (
                <option key={value} value={value}>
                  R$ {value.toLocaleString('pt-BR')}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Lote">
            <select className={selectClass} value={lot} onChange={(e) => setLot(asLot(e.target.value))} disabled={locked}>
              <option value="fixed">1 contrato</option>
              <option value="scaled">Crescente / R$ 1.000</option>
            </select>
          </Field>
          <Field label="Gráfico">
            <select className={selectClass} value={timeframe} onChange={(e) => setTimeframe(asTf(e.target.value))} disabled={locked}>
              <option value="m5">5 min</option>
              <option value="m1">1 min</option>
            </select>
          </Field>
          <Field label="De">
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
          </Field>
          <Field label="Até">
            <input
              type="date"
              className={selectClass}
              value={end}
              min={endMin}
              max={endMax || undefined}
              disabled={locked || source === 'mt5'}
              onChange={(e) => setEnd(e.target.value)}
            />
          </Field>
          <Field label="Fonte">
            <select
              className={selectClass}
              value={source}
              onChange={(e) => setSource(e.target.value as 'paper' | 'mt5')}
              disabled={snap.running}
            >
              <option value="paper">Paper (CSV)</option>
              <option value="mt5">MT5 (Windows)</option>
            </select>
          </Field>
          <Field label="Velocidade">
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
          </Field>
          <div className="flex h-9 items-center gap-2">
            <Button
              size="sm"
              disabled={busy || snap.running || Boolean(dateError)}
              onClick={() =>
                void post('/api/live/start', {
                  case: caseKey,
                  timeframe,
                  initial_bank: bank,
                  lot,
                  start: source === 'paper' ? start : undefined,
                  end: source === 'paper' ? end : undefined,
                  source,
                  interval_sec: intervalSec,
                })
              }
            >
              Iniciar{busy ? '…' : ''}
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

      <SessionDashboard snap={snap} emptyHint="Nenhuma ordem ainda. Inicie a sessão paper." />
    </div>
  )
}
