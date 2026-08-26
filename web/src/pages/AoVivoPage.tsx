import { useEffect, useRef, useState } from 'react'
import { SessionDashboard, SessionNav } from '@/components/SessionDashboard'
import { Button } from '@/components/ui/button'
import { EMPTY_SNAP, clock, isTodayStamp, type LiveSnap } from '@/lib/liveTypes'
import { readJson } from '@/lib/utils'

const ARMED_KEY = 'trader-ao-vivo-armed'

const WAIT_LABEL: Record<string, string> = {
  mercado_fechado: 'Armado. Paper começa no próximo pregão (09:15), com o M5 do feed',
  fora_do_ouro: 'Armado. Fora do horário de ouro (09:15–11:00 e 14:30–17:00)',
  fim_da_sessao: 'Armado. Paper volta no próximo pregão (09:15)',
  em_posicao: 'Posição paper aberta',
  aguardando_candle: 'Pregão aberto. Aguardando M5 (Yahoo ou demo WIN)',
  pronto: 'Pronto para simular no próximo sinal',
}

type ArmedStore = { armed: boolean; armed_at: string | null }

function localStamp() {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

function readArmed(): ArmedStore {
  try {
    const raw = localStorage.getItem(ARMED_KEY)
    if (!raw) return { armed: false, armed_at: null }
    const parsed = JSON.parse(raw) as ArmedStore
    return { armed: Boolean(parsed.armed), armed_at: parsed.armed_at || null }
  } catch {
    return { armed: false, armed_at: null }
  }
}

function writeArmed(armed: boolean, armed_at: string | null) {
  if (!armed) {
    localStorage.removeItem(ARMED_KEY)
    return
  }
  localStorage.setItem(ARMED_KEY, JSON.stringify({ armed: true, armed_at }))
}

function feedLabel(snap: LiveSnap) {
  const origin = snap.feed?.origin
  if (origin === 'yahoo') return 'yahoo · ^BVSP 5m'
  if (origin === 'demo') return 'demo · WIN$ relógio'
  if (origin === 'http' || snap.feed?.url) return 'URL · WIN_STREAM_URL'
  if (origin === 'ingest' && snap.feed?.ingested) return `ingest · ${snap.feed.ingested} candles`
  if (origin && origin !== 'none') return origin
  return 'aguardando M5'
}

export function AoVivoPage() {
  const stored = readArmed()
  const [snap, setSnap] = useState<LiveSnap>({
    ...EMPTY_SNAP,
    source: 'stream',
    order_mode: 'paper',
    interval_sec: 10,
    lot: 'scaled',
    running: stored.armed,
    armed_at: stored.armed_at,
  })
  const [offline, setOffline] = useState(false)
  const [busy, setBusy] = useState(false)
  const restoring = useRef(false)
  const wasOffline = useRef(false)

  const applySnap = (json: LiveSnap, keepArmed = false) => {
    const saved = readArmed()
    const running = Boolean(json.running || (keepArmed && saved.armed))
    if (running) {
      writeArmed(true, json.armed_at || saved.armed_at || localStamp())
    }
    setSnap({
      ...json,
      running,
      armed_at: json.armed_at || saved.armed_at,
    })
    setOffline(false)
  }

  const pull = async () => {
    try {
      const saved = readArmed()
      const q =
        saved.armed && saved.armed_at ? `?armed_at=${encodeURIComponent(saved.armed_at)}` : ''
      const res = await fetch(`/api/realtime${q}`, { cache: 'no-store' })
      if (!res.ok) {
        const json = await readJson<{ detail?: string }>(res)
        throw new Error(typeof json.detail === 'string' ? json.detail : 'API ao vivo indisponível')
      }
      const json = await readJson<LiveSnap>(res)
      const reconnect = wasOffline.current
      wasOffline.current = false
      applySnap(json, saved.armed)
      if ((reconnect || (saved.armed && !json.running)) && saved.armed && !restoring.current) {
        restoring.current = true
        void post('/api/realtime/start', { order_mode: 'paper', armed_at: saved.armed_at }).finally(() => {
          restoring.current = false
        })
      }
    } catch (err) {
      wasOffline.current = true
      setOffline(true)
      setSnap((prev) => ({
        ...prev,
        error: err instanceof Error ? err.message : 'API offline',
      }))
    }
  }

  useEffect(() => {
    void pull()
    const id = window.setInterval(() => {
      void pull()
    }, 2000)
    return () => window.clearInterval(id)
  }, [])

  const post = async (path: string, body?: object) => {
    setBusy(true)
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        cache: 'no-store',
      })
      const json = await readJson<LiveSnap & { detail?: string }>(res)
      if (!res.ok) throw new Error(typeof json.detail === 'string' ? json.detail : 'falhou')
      if (path.includes('/stop') || path.includes('/reset')) {
        writeArmed(false, null)
      } else if (json.running) {
        writeArmed(true, json.armed_at || readArmed().armed_at)
      }
      applySnap(json)
    } catch (err) {
      setSnap((prev) => ({ ...prev, error: err instanceof Error ? err.message : 'falhou' }))
    } finally {
      setBusy(false)
    }
  }

  const arm = () => {
    const armedAt = readArmed().armed_at || localStamp()
    writeArmed(true, armedAt)
    setSnap((prev) => ({ ...prev, running: true, armed_at: armedAt, error: null }))
    void post('/api/realtime/start', { order_mode: 'paper', armed_at: armedAt })
  }

  const pause = () => {
    writeArmed(false, null)
    setSnap((prev) => ({ ...prev, running: false }))
    void post('/api/realtime/stop')
  }

  const wait = snap.wait_reason || 'aguardando_candle'
  const status = offline
    ? 'API offline — rode localmente: python -m trader serve'
    : !snap.running
      ? 'Pausado — o motor não está armado'
      : WAIT_LABEL[wait] || wait

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <SessionNav />

      <section className="relative mx-auto max-w-6xl px-5 pb-4 sm:px-8">
        <p className="text-sm uppercase tracking-[0.2em] text-primary">Ao vivo · simular</p>
        <h1 className="mt-3 font-display text-4xl font-bold">Operação em tempo real</h1>
        <p className="mt-2 max-w-3xl text-muted-foreground">
          {status}. 5 min · lote crescente / R$ 1.000 · best_candles_m5_1000_a
          {isTodayStamp(snap.last_bar_time) ? ` · candle ${clock(snap.last_bar_time)}` : ''}
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="rounded-lg bg-gain/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-gain">
            SIMULAR · paper
          </span>
          <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">{feedLabel(snap)}</span>
          {snap.feed?.detail ? (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">{snap.feed.detail}</span>
          ) : null}
          {snap.running ? (
            <span className="rounded-lg bg-primary/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-primary">
              armado
            </span>
          ) : (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">pausado</span>
          )}
          {snap.next_gold ? (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
              próximo ouro {clock(snap.next_gold)}
            </span>
          ) : (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">dentro do ouro</span>
          )}
        </div>
        <div className="mt-4 flex h-9 flex-wrap items-center gap-2">
          <Button size="sm" disabled={busy || snap.running} onClick={arm}>
            {snap.running ? 'Armado' : busy ? 'Armar…' : 'Armar'}
          </Button>
          <Button variant="outline" size="sm" disabled={busy || !snap.running} onClick={pause}>
            Pausar
          </Button>
          <Button variant="ghost" size="sm" disabled={busy} onClick={() => void post('/api/realtime/reset')}>
            Zerar
          </Button>
        </div>
        {snap.error && !snap.error.startsWith('Stream sem candles') ? (
          <p className="mt-3 text-sm text-loss">{snap.error}</p>
        ) : null}
      </section>

      <SessionDashboard
        snap={snap}
        emptyHint="Nenhuma ordem paper ainda. Armado espera o ouro e um M5 novo (Yahoo ^BVSP ou demo WIN no relógio)."
      />
    </div>
  )
}
