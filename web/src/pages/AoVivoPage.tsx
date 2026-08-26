import { useEffect, useRef, useState } from 'react'
import { SessionDashboard, SessionNav } from '@/components/SessionDashboard'
import { Button } from '@/components/ui/button'
import { EMPTY_SNAP, clock, isTodayStamp, type LiveSnap } from '@/lib/liveTypes'
import { apiFetch, brl, cn, readJson } from '@/lib/utils'

const WAIT_LABEL: Record<string, string> = {
  aguardando_login: 'Aguardando o terminal MT5 Genial',
  conta_real: 'Conta real — simulando paper, sem envio no MT5',
  sem_simbolo: 'Sem WIN negociável no Market Watch (WINV26 + WIN$)',
  autotrading_desligado: 'AutoTrading desligado — o motor tenta religar sozinho no terminal Genial',
  mercado_fechado: 'ARMADO · fora do pregão. Espera o próximo ouro (09:15) e opera sozinho',
  fora_do_ouro: 'ARMADO · almoço / fora do ouro. Espera 09:15–11:00 ou 14:30–17:00 e opera sozinho',
  fim_da_sessao: 'Sessão encerrada às 17:00',
  em_posicao: 'Posição aberta',
  aguardando_candle: 'Pregão aberto. Aguardando M5 do MT5',
  pronto: 'Pregão aberto. Aguardando o próximo sinal',
}

type OrderMode = 'paper' | 'mt5' | 'prd'

function isOrderMode(value: unknown): value is OrderMode {
  return value === 'paper' || value === 'mt5' || value === 'prd'
}

function realBank(mt5?: LiveSnap['mt5'] | null) {
  if (!mt5) return null
  const raw = mt5.bank ?? mt5.equity ?? mt5.balance ?? mt5.margin_free
  return raw == null ? null : Number(raw)
}

function pushLog(line: string, extra?: unknown) {
  if (line === 'snap_not_running' || line === 'poll_fail') return
  if (line === 'window_error' && String(extra).includes('out of memory')) return
  console.log('[ao-vivo]', line, extra ?? '')
  void apiFetch('/api/realtime/ui-log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ line, extra: extra ?? null }),
  }).catch((err) => console.warn('[ao-vivo] log fail', err))
}

export function AoVivoPage() {
  const [snap, setSnap] = useState<LiveSnap>({
    ...EMPTY_SNAP,
    source: 'mt5',
    order_mode: 'paper',
    interval_sec: 1,
    lot: 'scaled',
  })
  const [orderMode, setOrderMode] = useState<OrderMode>('paper')
  const [offline, setOffline] = useState(false)
  const [busy, setBusy] = useState(false)
  const [bankDialog, setBankDialog] = useState<{
    amount: number
    login: number | null
    server: string | null
  } | null>(null)
  const pendingMode = useRef<OrderMode | null>(null)
  const booted = useRef(false)
  const busyRef = useRef(false)
  const pullingRef = useRef(false)
  const wantArmed = useRef<boolean | null>(null)
  const toldEmptyBank = useRef(false)

  const applySnap = (json: LiveSnap, from: string) => {
    const next = { ...json }
    if (wantArmed.current != null && from === 'poll') {
      next.running = wantArmed.current
    }
    if (wantArmed.current != null && from !== 'poll' && Boolean(next.running) === wantArmed.current) {
      wantArmed.current = null
    }
    setSnap(next)
    const armed = Boolean(next.running || next.armed)
    if (!armed && from !== 'boot') {
      pushLog(`snap_not_running from=${from}`, {
        wait: json.wait_reason,
        mode: json.order_mode,
        err: json.error,
      })
    }
    if (pendingMode.current) return
    if (isOrderMode(json.order_mode)) setOrderMode(json.order_mode)
    else if (isOrderMode(json.mode)) setOrderMode(json.mode)
    setOffline(false)
  }

  const pull = async () => {
    if (busyRef.current || pullingRef.current) return
    pullingRef.current = true
    try {
      const res = await apiFetch('/api/realtime')
      if (!res.ok) {
        const json = await readJson<{ detail?: string }>(res)
        throw new Error(typeof json.detail === 'string' ? json.detail : 'API ao vivo indisponível')
      }
      const json = await readJson<LiveSnap>(res)
      applySnap(json, 'poll')
    } catch (err) {
      setOffline(true)
      setSnap((prev) => ({
        ...prev,
        error: err instanceof Error ? err.message : 'API offline',
      }))
    } finally {
      pullingRef.current = false
    }
  }

  const post = async (path: string, body?: object) => {
    busyRef.current = true
    setBusy(true)
    pushLog(`post ${path}`, body)
    try {
      const res = await apiFetch(path, {
        method: 'POST',
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      })
      const json = await readJson<LiveSnap & { detail?: string }>(res)
      if (!res.ok) throw new Error(typeof json.detail === 'string' ? json.detail : 'falhou')
      pushLog('post_ok', { path, running: json.running, mode: json.order_mode, wait: json.wait_reason })
      if (pendingMode.current && (json.order_mode === pendingMode.current || json.mode === pendingMode.current)) {
        pendingMode.current = null
      }
      if (pendingMode.current === 'mt5' && json.order_mode === 'paper') {
        pendingMode.current = null
      }
      applySnap(json, path)
      if (!pendingMode.current) setOffline(false)
    } catch (err) {
      pendingMode.current = null
      const message = err instanceof Error ? err.message : 'falhou'
      pushLog('post_fail', { path, message })
      setSnap((prev) => ({ ...prev, error: message }))
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  useEffect(() => {
    const onErr = (event: ErrorEvent) => pushLog('window_error', event.message)
    const onRej = (event: PromiseRejectionEvent) => pushLog('unhandled', String(event.reason))
    window.addEventListener('error', onErr)
    window.addEventListener('unhandledrejection', onRej)
    const boot = async () => {
      try {
        const res = await apiFetch('/api/realtime')
        const json = await readJson<LiveSnap>(res)
        applySnap(json, 'boot')
        pushLog('boot', {
          running: json.running,
          mode: json.order_mode,
          wait: json.wait_reason,
          demo: json.mt5?.demo,
          href: window.location.href,
        })
        if (!json.running && !booted.current) {
          booted.current = true
          const mode: OrderMode = json.order_mode === 'prd' ? 'paper' : json.mt5?.demo === false ? 'paper' : 'mt5'
          await post('/api/realtime/start', { order_mode: mode, source: 'mt5' })
        }
      } catch (err) {
        pushLog('boot_fail', err instanceof Error ? err.message : err)
        setOffline(true)
      }
    }
    void boot()
    const id = window.setInterval(() => {
      void pull()
    }, 2000)
    return () => {
      window.removeEventListener('error', onErr)
      window.removeEventListener('unhandledrejection', onRej)
      window.clearInterval(id)
    }
  }, [])

  const changeOrder = (next: OrderMode) => {
    pendingMode.current = next
    setOrderMode(next)
    void post('/api/realtime/source', { order_mode: next, source: 'mt5' })
  }

  useEffect(() => {
    const amount = realBank(snap.mt5)
    if (snap.mt5?.login == null || amount == null) return
    if (amount <= 0 && !toldEmptyBank.current) {
      toldEmptyBank.current = true
      setBankDialog({ amount, login: snap.mt5.login, server: snap.mt5.server })
    }
    if (amount > 0) toldEmptyBank.current = false
  }, [snap.mt5])

  const armNow = () => {
    if (orderMode === 'prd') {
      const ok = window.confirm(
        'Modo Produção: no próximo sinal válido (ouro, não FLAT) a API chama order_send nesta conta real Genial, com stop e alvo. Continuar?',
      )
      if (!ok) return
    }
    pendingMode.current = null
    wantArmed.current = true
    setSnap((prev) => ({ ...prev, running: true, armed: true, error: null }))
    void post('/api/realtime/start', { order_mode: orderMode, source: 'mt5' })
  }

  const pauseNow = () => {
    wantArmed.current = false
    setSnap((prev) => ({ ...prev, running: false, armed: false }))
    void post('/api/realtime/stop')
  }

  const mt5 = snap.mt5
  const wait = snap.wait_reason || 'aguardando_candle'
  const paper = orderMode === 'paper'
  const prod = orderMode === 'prd'
  const armed = Boolean(snap.running || snap.armed)
  const liveBank = realBank(mt5)
  const status = offline
    ? 'API offline — rode localmente: python -m trader serve'
    : !armed
      ? 'Pausado — clique Armar'
      : WAIT_LABEL[wait] || wait

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <SessionNav />

      <section className="relative mx-auto max-w-6xl px-5 pb-4 sm:px-8">
        <p className="text-sm uppercase tracking-[0.2em] text-primary">
          {paper ? 'Ao vivo · simular' : prod ? 'Ao vivo · produção PRD' : 'Ao vivo · enviar MT5'}
        </p>
        <h1 className="mt-3 font-display text-4xl font-bold">Operação em tempo real</h1>
        <p className="mt-2 max-w-3xl text-muted-foreground">
          {status}. 5 min · lote 1 mini a partir de R$ 500 · +1 / R$ 1.000 · best_candles_m5_1000_a
          {snap.last_tick ? ` · tick ${clock(snap.last_tick)}` : ''}
          {isTodayStamp(snap.last_bar_time) ? ` · candle ${clock(snap.last_bar_time)}` : ''}
          {snap.feed?.detail ? ` · ${snap.feed.detail}` : ''}
          {snap.quote ? ` · último ${snap.quote.last.toFixed(0)}` : ''}
          {snap.position && snap.open_pnl != null ? ` · P&L aberto ${brl(snap.open_pnl)}` : ''}
          {snap.signal?.reason ? ` · sinal ${snap.signal.side} (${snap.signal.reason})` : ''}
        </p>
        <p className="mt-3 max-w-3xl text-sm text-muted-foreground">
          Os três modos leem o WIN no MT5 (tick + M5) em tempo real. <strong>Simular</strong> nunca chama order_send:
          marca compra/venda local e calcula P&L para estudo. <strong>Enviar</strong> manda ordem só na demo; no PRD
          continua paper. <strong>Produção</strong> envia de fato na conta real no sinal válido (ouro, stop/alvo, lote
          crescente). AutoTrading precisa estar ligado.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span
            className={
              armed
                ? 'rounded-lg bg-gain/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-gain'
                : 'rounded-lg bg-loss/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-loss'
            }
          >
            {armed ? 'ARMADO' : 'PAUSADO'}
          </span>
          <span
            className={
              prod
                ? 'rounded-lg bg-loss/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-loss'
                : 'rounded-lg bg-gain/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-gain'
            }
          >
            {paper
              ? 'SIMULAR · paper local · feed MT5'
              : prod
                ? 'PRODUÇÃO · order_send na conta real'
                : mt5?.demo === true
                  ? 'ENVIAR · order_send na demo'
                  : 'ENVIAR · conta real · paper até a demo'}
          </span>
          <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
            {mt5?.server || 'sem servidor'} · {mt5?.login ?? 'sem login'}
          </span>
          <span className="rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground">
            {mt5?.symbol || 'sem símbolo'} {mt5?.filling ? `· ${mt5.filling}` : ''}
          </span>
          <span
            className={
              mt5?.trade_allowed
                ? 'rounded-lg bg-gain/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-gain'
                : 'rounded-lg bg-loss/15 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-loss'
            }
          >
            {mt5?.trade_allowed ? 'AutoTrading on' : 'AutoTrading off'}
          </span>
          <span
            className={
              liveBank != null && liveBank > 0
                ? 'rounded-lg bg-gain/15 px-2 py-1 text-xs font-semibold tabular-nums text-gain'
                : 'rounded-lg bg-loss/15 px-2 py-1 text-xs font-semibold tabular-nums text-loss'
            }
          >
            Banca real {liveBank == null ? '—' : brl(liveBank)}
          </span>
          {snap.quote ? (
            <span className="rounded-lg border border-border px-2 py-1 text-xs tabular-nums text-muted-foreground">
              {snap.quote.bid.toFixed(0)} / {snap.quote.ask.toFixed(0)}
            </span>
          ) : null}
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
              <option value="prd">Produção (conta real)</option>
            </select>
          </label>
          <div className="flex h-9 items-center gap-2">
            <div className="inline-flex h-9 overflow-hidden rounded-lg border border-border bg-elevated/60">
              <button
                type="button"
                aria-pressed={armed}
                disabled={busy}
                onClick={() => armNow()}
                className={cn(
                  'h-9 px-4 text-sm font-semibold transition-colors',
                  armed
                    ? 'bg-gain text-black'
                    : 'text-muted-foreground hover:bg-card',
                )}
              >
                {armed ? 'Armado' : busy ? 'Armando…' : 'Armar'}
              </button>
              <button
                type="button"
                aria-pressed={!armed}
                disabled={busy || !armed}
                onClick={() => pauseNow()}
                className={cn(
                  'h-9 px-4 text-sm font-semibold transition-colors',
                  !armed
                    ? 'bg-card text-foreground'
                    : 'text-muted-foreground hover:bg-card',
                )}
              >
                {armed ? 'Pausar' : 'Pausado'}
              </button>
            </div>
            <Button variant="ghost" size="sm" disabled={busy} onClick={() => void post('/api/realtime/reset')}>
              Zerar
            </Button>
          </div>
        </div>
        {orderMode === 'mt5' && mt5?.demo === false ? (
          <p className="mt-3 text-sm text-loss">
            Enviar só na demo. Nesta conta o motor simula paper e não chama order_send. Para enviar no PRD, use Produção.
          </p>
        ) : null}
        {prod ? (
          <p className="mt-3 text-sm text-loss">
            Produção envia ordem real no Genial PRD no próximo sinal válido (09:15–11:00 e 14:30–17:00, não FLAT). O
            motor mantém o AutoTrading ligado.
          </p>
        ) : null}
        {snap.error && !snap.error.startsWith('Stream sem candles') ? (
          <p className="mt-3 text-sm text-loss">{snap.error}</p>
        ) : null}
        {snap.skip_reason ? <p className="mt-3 text-sm text-amber-400">{snap.skip_reason}</p> : null}
        {wait === 'aguardando_login' && snap.playbook ? (
          <pre className="mt-4 max-w-3xl overflow-x-auto whitespace-pre-wrap rounded-2xl border border-border bg-elevated/50 p-4 text-sm text-muted-foreground">
            {snap.playbook}
          </pre>
        ) : null}
      </section>

      {bankDialog ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-5">
          <div className="w-full max-w-md rounded-2xl border border-border bg-elevated p-6 shadow-xl">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Conta real</p>
            <h2 className="mt-2 font-display text-2xl font-bold">Banca no MT5</h2>
            <p className="mt-3 text-muted-foreground">
              {bankDialog.server || 'Genial'} · {bankDialog.login ?? 'sem login'}
            </p>
            <p className="mt-4 font-display text-4xl font-bold tabular-nums">
              {brl(bankDialog.amount)}
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              {bankDialog.amount <= 0
                ? 'A conta real está sem banca (0). O ao vivo continua espelhando o valor do terminal.'
                : 'Este é o saldo/equity lido agora no terminal.'}
            </p>
            <Button className="mt-6 w-full" onClick={() => setBankDialog(null)}>
              Entendi
            </Button>
          </div>
        </div>
      ) : null}

      <SessionDashboard
        snap={snap}
        emptyHint={
          wait === 'fora_do_ouro'
            ? `Almoço. Sem ordem até o ouro ${snap.next_gold ? clock(snap.next_gold) : '14:30'}. Os sinais já aparecem acima; a tabela Ordens preenche no primeiro fill.`
            : wait === 'mercado_fechado'
              ? `Fora do pregão. Sem ordem até o ouro ${snap.next_gold ? clock(snap.next_gold) : '09:15'}.`
              : paper
                ? 'Nenhuma ordem paper ainda. Armado lê o M5 do WIN no MT5 e simula o P&L. Nenhuma ordem é enviada.'
                : prod
                  ? 'Nenhuma ordem real ainda. No ouro, um sinal BUY/SELL válido chama order_send nesta conta PRD.'
                  : 'Nenhuma ordem enviada. O motor lê o M5 do WIN no MT5 e envia na demo no próximo sinal, se estiver flat.'
        }
      />
    </div>
  )
}
