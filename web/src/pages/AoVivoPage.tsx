import { useEffect, useRef, useState } from 'react'
import { SessionDashboard, SessionNav } from '@/components/SessionDashboard'
import { Button } from '@/components/ui/button'
import { EMPTY_SNAP, clock, isTodayStamp, type LiveSnap } from '@/lib/liveTypes'
import { brl, readJson } from '@/lib/utils'

const WAIT_LABEL: Record<string, string> = {
  aguardando_login: 'Aguardando login demo no MT5',
  conta_real: 'Conta real detectada — o robô não envia ordem',
  sem_simbolo: 'Sem WIN negociável no Market Watch',
  autotrading_desligado: 'Ligue o AutoTrading no MT5',
  mercado_fechado: 'Armado. Paper começa no próximo ouro (09:15), com o candle do stream. Sem ordem no MT5',
  fora_do_ouro: 'Fora do horário de ouro (09:15–11:00 e 14:30–17:00)',
  fim_da_sessao: 'Sessão encerrada às 17:00',
  em_posicao: 'Posição paper aberta',
  aguardando_candle: 'Pregão aberto. Aguardando M5 do stream',
  pronto: 'Pronto para simular no próximo sinal',
}

type OrderMode = 'paper' | 'mt5'

export function AoVivoPage() {
  const [snap, setSnap] = useState<LiveSnap>({ ...EMPTY_SNAP, source: 'stream', order_mode: 'paper', interval_sec: 10, lot: 'scaled' })
  const [orderMode, setOrderMode] = useState<OrderMode>('paper')
  const [offline, setOffline] = useState(false)
  const [busy, setBusy] = useState(false)
  const pendingMode = useRef<OrderMode | null>(null)

  const applySnap = (json: LiveSnap) => {
    setSnap(json)
    if (pendingMode.current) return
    if (json.order_mode === 'paper' || json.order_mode === 'mt5') setOrderMode(json.order_mode)
    else if (json.mode === 'paper' || json.mode === 'mt5') setOrderMode(json.mode)
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
      applySnap(json)
    } catch (err) {
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
      if (pendingMode.current && (json.order_mode === pendingMode.current || json.mode === pendingMode.current)) {
        pendingMode.current = null
      }
      applySnap(json)
      if (!pendingMode.current) setOffline(false)
    } catch (err) {
      pendingMode.current = null
      setSnap((prev) => ({ ...prev, error: err instanceof Error ? err.message : 'falhou' }))
    } finally {
      setBusy(false)
    }
  }

  const changeOrder = (next: OrderMode) => {
    pendingMode.current = next
    setOrderMode(next)
    void post('/api/realtime/source', { order_mode: next, source: next === 'paper' ? 'stream' : 'mt5' })
  }

  const mt5 = snap.mt5
  const wait = snap.wait_reason || 'aguardando_candle'
  const mt5Down = orderMode === 'mt5' && !mt5?.ready
  const paper = orderMode === 'paper'
  const status = offline
    ? 'API offline — rode localmente: python -m trader serve'
    : !snap.running
      ? 'Pausado — o motor não está armado'
      : paper && wait === 'mercado_fechado'
        ? WAIT_LABEL.mercado_fechado
        : paper && wait === 'aguardando_candle'
          ? WAIT_LABEL.aguardando_candle
          : WAIT_LABEL[wait] || wait

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <SessionNav />

      <section className="relative mx-auto max-w-6xl px-5 pb-4 sm:px-8">
        <p className="text-sm uppercase tracking-[0.2em] text-primary">
          {paper ? 'Ao vivo · simular' : 'Ao vivo · enviar MT5'}
        </p>
        <h1 className="mt-3 font-display text-4xl font-bold">Operação em tempo real</h1>
        <p className="mt-2 max-w-3xl text-muted-foreground">
          {status}. 5 min · lote crescente / R$ 1.000 · best_candles_m5_1000_a
          {isTodayStamp(snap.last_bar_time) ? ` · candle ${clock(snap.last_bar_time)}` : ''}
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="rounded-lg bg-gain/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-gain">
            {paper ? 'SIMULAR · paper' : mt5?.demo === false ? 'REAL (bloqueado)' : 'ENVIAR · MT5'}
          </span>
          {paper ? (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
              stream · {snap.feed?.url ? 'URL' : snap.feed?.ingested ? `${snap.feed.ingested} ingest` : 'aguardando M5 de hoje'}
            </span>
          ) : (
            <>
              <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
                {mt5?.server || 'sem servidor'} · {mt5?.login ?? 'sem login'}
              </span>
              <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
                {mt5?.symbol || 'sem símbolo'} {mt5?.filling ? `· ${mt5.filling}` : ''}
              </span>
              {mt5?.equity != null ? (
                <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
                  equity {brl(mt5.equity)}
                </span>
              ) : null}
            </>
          )}
          {snap.next_gold ? (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
              próximo ouro {clock(snap.next_gold)}
            </span>
          ) : (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">dentro do ouro</span>
          )}
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-2">
          <label className="flex min-w-0 flex-col justify-end gap-1 text-[11px] uppercase tracking-wide text-muted-foreground">
            Ordem
            <select
              className="h-9 rounded-lg border border-border bg-elevated/60 px-3 text-sm"
              value={orderMode}
              onChange={(e) => changeOrder(e.target.value as OrderMode)}
            >
              <option value="paper">Simular</option>
              <option value="mt5">Enviar (MT5 demo)</option>
            </select>
          </label>
          <div className="flex h-9 items-center gap-2">
            <Button
              size="sm"
              disabled={busy || snap.running}
              onClick={() => void post('/api/realtime/start', { order_mode: orderMode, source: paper ? 'stream' : 'mt5' })}
            >
              Armar{busy ? '…' : ''}
            </Button>
            <Button variant="outline" size="sm" disabled={busy || !snap.running} onClick={() => void post('/api/realtime/stop')}>
              Pausar
            </Button>
            <Button variant="ghost" size="sm" disabled={busy} onClick={() => void post('/api/realtime/reset')}>
              Zerar
            </Button>
          </div>
        </div>
        {mt5Down ? (
          <p className="mt-3 text-sm text-loss">MT5 fora do ar. Enviar ordem fica bloqueado até o login demo. Use Simular para o teste.</p>
        ) : null}
        {snap.error && !snap.error.startsWith('Stream sem candles') ? (
          <p className="mt-3 text-sm text-loss">{snap.error}</p>
        ) : null}
        {orderMode === 'mt5' && wait === 'aguardando_login' && snap.playbook ? (
          <pre className="mt-4 max-w-3xl overflow-x-auto whitespace-pre-wrap rounded-2xl border border-border bg-elevated/50 p-4 text-sm text-muted-foreground">
            {snap.playbook}
          </pre>
        ) : null}
      </section>

      <SessionDashboard
        snap={snap}
        emptyHint={
          paper
            ? 'Nenhuma ordem paper ainda. Armado espera o ouro e o M5 do stream (WIN_STREAM_URL ou POST /api/realtime/candles).'
            : 'Nenhuma ordem enviada. O motor espera o horário de ouro, o login demo e um candle M5 novo.'
        }
      />
    </div>
  )
}
