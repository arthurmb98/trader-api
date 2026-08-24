import { useEffect, useRef, useState } from 'react'
import { SessionDashboard, SessionNav } from '@/components/SessionDashboard'
import { Button } from '@/components/ui/button'
import { EMPTY_SNAP, clock, isTodayStamp, type LiveSnap } from '@/lib/liveTypes'
import { readJson } from '@/lib/utils'

const WAIT_LABEL: Record<string, string> = {
  mercado_fechado: 'Armado. Paper começa no próximo ouro (09:15), com o M5 do feed',
  fora_do_ouro: 'Fora do horário de ouro (09:15–11:00 e 14:30–17:00)',
  fim_da_sessao: 'Sessão encerrada às 17:00',
  em_posicao: 'Posição paper aberta',
  aguardando_candle: 'Pregão aberto. Aguardando M5 (Yahoo ou demo WIN)',
  pronto: 'Pronto para simular no próximo sinal',
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
  const [snap, setSnap] = useState<LiveSnap>({
    ...EMPTY_SNAP,
    source: 'stream',
    order_mode: 'paper',
    interval_sec: 10,
    lot: 'scaled',
  })
  const [offline, setOffline] = useState(false)
  const [busy, setBusy] = useState(false)
  const wasOffline = useRef(false)

  const applySnap = (json: LiveSnap) => {
    setSnap(json)
    setOffline(false)
  }

  const pull = async () => {
    try {
      const res = await fetch('/api/realtime', { cache: 'no-store' })
      if (!res.ok) {
        const json = await readJson<{ detail?: string }>(res)
        throw new Error(typeof json.detail === 'string' ? json.detail : 'API ao vivo indisponível')
      }
      const json = await readJson<LiveSnap>(res)
      const reconnect = wasOffline.current
      wasOffline.current = false
      applySnap(json)
      if (reconnect && !json.running) {
        void post('/api/realtime/start', { order_mode: 'paper' })
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
      })
      const json = await readJson<LiveSnap & { detail?: string }>(res)
      if (!res.ok) throw new Error(typeof json.detail === 'string' ? json.detail : 'falhou')
      applySnap(json)
    } catch (err) {
      setSnap((prev) => ({ ...prev, error: err instanceof Error ? err.message : 'falhou' }))
    } finally {
      setBusy(false)
    }
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
          {snap.next_gold ? (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
              próximo ouro {clock(snap.next_gold)}
            </span>
          ) : (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">dentro do ouro</span>
          )}
        </div>
        <div className="mt-4 flex h-9 flex-wrap items-center gap-2">
          <Button size="sm" disabled={busy || snap.running} onClick={() => void post('/api/realtime/start', { order_mode: 'paper' })}>
            Armar{busy ? '…' : ''}
          </Button>
          <Button variant="outline" size="sm" disabled={busy || !snap.running} onClick={() => void post('/api/realtime/stop')}>
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
