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
  mercado_fechado: 'Fora do pregão — operação só no horário de hoje',
  fora_do_ouro: 'Fora do horário de ouro (09:15–11:00 e 14:30–17:00)',
  fim_da_sessao: 'Sessão encerrada às 17:00',
  em_posicao: 'Posição aberta',
  aguardando_candle: 'Esperando o próximo candle M5 fechado',
  pronto: 'Pronto para operar no próximo sinal',
}

type FeedKey = 'mt5' | 'stream'

export function AoVivoPage() {
  const [snap, setSnap] = useState<LiveSnap>({ ...EMPTY_SNAP, source: 'mt5', interval_sec: 10 })
  const [source, setSource] = useState<FeedKey>('mt5')
  const [offline, setOffline] = useState(false)
  const [busy, setBusy] = useState(false)
  const pendingSource = useRef<FeedKey | null>(null)

  const applySnap = (json: LiveSnap) => {
    setSnap(json)
    if (pendingSource.current) return
    if (json.source === 'mt5' || json.source === 'stream') setSource(json.source)
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
      if (pendingSource.current && json.source === pendingSource.current) pendingSource.current = null
      applySnap(json)
      if (!pendingSource.current) setOffline(false)
    } catch (err) {
      pendingSource.current = null
      setSnap((prev) => ({ ...prev, error: err instanceof Error ? err.message : 'falhou' }))
    } finally {
      setBusy(false)
    }
  }

  const changeSource = (next: FeedKey) => {
    pendingSource.current = next
    setSource(next)
    void post('/api/realtime/source', { source: next })
  }

  const mt5 = snap.mt5
  const wait = snap.wait_reason || 'aguardando_login'
  const mt5Down = source === 'mt5' && !mt5?.ready
  const armarOff = busy
    ? 'Armar off: POST em andamento (busy)'
    : snap.running
      ? 'Armar off: motor já armado (running=true). Use Pausar.'
      : offline
        ? 'Armar off: API offline'
        : 'Armar livre'
  const status = offline
    ? 'API offline — rode localmente: python -m trader serve'
    : snap.running
      ? WAIT_LABEL[wait] || wait
      : 'Pausado — o motor não está armado'

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <SessionNav />

      <section className="relative mx-auto max-w-6xl px-5 pb-4 sm:px-8">
        <p className="text-sm uppercase tracking-[0.2em] text-primary">
          {source === 'stream' ? 'Ao vivo · stream paper' : 'Ao vivo · MT5 demo'}
        </p>
        <h1 className="mt-3 font-display text-4xl font-bold">Operação em tempo real</h1>
        <p className="mt-2 max-w-3xl text-muted-foreground">
          {status}. 5 min · 1 mini · best_candles_m5_1000_a
          {isTodayStamp(snap.last_bar_time) ? ` · candle ${clock(snap.last_bar_time)}` : ' · sem candle de hoje'}
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="rounded-lg bg-gain/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-gain">
            {source === 'stream' ? 'STREAM · paper' : mt5?.demo === false ? 'REAL (bloqueado)' : 'DEMO · MT5'}
          </span>
          {source === 'mt5' ? (
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
          ) : (
            <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
              {snap.feed?.symbol || 'WIN'} · {snap.feed?.file || snap.feed?.detail || 'aguardando candles'}
            </span>
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
            Fonte
            <select
              className="h-9 rounded-lg border border-border bg-elevated/60 px-3 text-sm"
              value={source}
              onChange={(e) => changeSource(e.target.value as FeedKey)}
            >
              <option value="mt5">MetaTrader 5</option>
              <option value="stream">Stream (paper)</option>
            </select>
          </label>
          <div className="flex h-9 items-center gap-2">
            <Button size="sm" disabled={busy || snap.running} onClick={() => void post('/api/realtime/start', { source })}>
              Armar{busy ? '…' : ''}
            </Button>
            <Button variant="outline" size="sm" disabled={busy || !snap.running} onClick={() => void post('/api/realtime/stop')}>
              Pausar
            </Button>
            <Button variant="ghost" size="sm" disabled={busy} onClick={() => void post('/api/realtime/reset')}>
              Zerar
            </Button>
          </div>
          <p className="w-full font-mono text-[11px] text-muted-foreground">
            debug · {armarOff} · running={String(snap.running)} · busy={String(busy)} · wait={wait} ·
            source={source}
          </p>
        </div>
        {mt5Down ? (
          <p className="mt-3 text-sm text-loss">MT5 fora do ar. Escolha Stream para continuar simulando.</p>
        ) : null}
        {snap.error && !snap.error.startsWith('Stream sem candles') ? (
          <p className="mt-3 text-sm text-loss">{snap.error}</p>
        ) : null}
        {source === 'mt5' && wait === 'aguardando_login' && snap.playbook ? (
          <pre className="mt-4 max-w-3xl overflow-x-auto whitespace-pre-wrap rounded-2xl border border-border bg-elevated/50 p-4 text-sm text-muted-foreground">
            {snap.playbook}
          </pre>
        ) : null}
      </section>

      <SessionDashboard
        snap={snap}
        emptyHint={
          source === 'stream'
            ? snap.feed?.origin === 'file'
              ? 'CSV antigo só no gráfico. Ordem paper só com candle de hoje (POST /candles, URL ou MT5).'
              : 'Nenhuma ordem paper ainda. Só candle de hoje, no horário de ouro.'
            : 'Nenhuma ordem ao vivo ainda. O motor espera o horário de ouro e um candle M5 novo.'
        }
      />
    </div>
  )
}
