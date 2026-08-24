export type SignalSnap = {
  side: string
  entry: number
  stop: number
  take: number
  reason: string
  predicted?: { abertura: number; maximo: number; minimo: number; fechamento: number } | null
}

export type TradeSnap = {
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

export type CaseKey = 'last_candle' | 'last_candles'
export type TfKey = 'm1' | 'm5'
export type LotKey = 'fixed' | 'scaled'
export type PeriodLevel = 'daily' | 'weekly' | 'monthly' | 'quarterly'

export type PeriodAvg = {
  avg: number
  avg_gain: number
  avg_loss: number
  n: number
  n_gain: number
  n_loss: number
}

export type PeriodStats = {
  window_days: number
  levels: PeriodLevel[]
  series: Partial<Record<PeriodLevel, { t: string; pnl: number }[]>>
  avg: Partial<Record<PeriodLevel, PeriodAvg>>
}

export type Mt5Snap = {
  ready: boolean
  demo: boolean | null
  login: number | null
  server: string | null
  symbol: string | null
  filling?: string | null
  trade_allowed: boolean
  balance: number | null
  equity: number | null
}

export type LiveSnap = {
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
  lot?: LotKey | string
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
  wait_reason?: string
  next_gold?: string | null
  playbook?: string | null
  mode?: 'paper' | 'mt5' | string
  feed?: {
    ready?: boolean
    symbol?: string | null
    detail?: string | null
    error?: string | null
    origin?: string | null
    file?: string | null
  }
  mt5?: Mt5Snap
}

export type LiveMeta = {
  banks: number[]
  cases: { key: CaseKey; label: string }[]
  timeframes: { key: TfKey; label: string }[]
  timeframe: TfKey
  min_date: string
  max_date: string
  default_start: string
  default_end: string
  max_span_months: number
  lots?: { key: LotKey; label: string }[]
  lot?: LotKey
}

export const EMPTY_SNAP: LiveSnap = {
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
  lot: 'fixed',
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

export const CASE_LABEL: Record<CaseKey, string> = {
  last_candle: 'Último candle',
  last_candles: 'Últimos candles',
}
export const TF_LABEL: Record<TfKey, string> = { m1: '1 min', m5: '5 min' }
export const LOT_LABEL: Record<LotKey, string> = {
  fixed: '1 contrato',
  scaled: 'Crescente / R$ 1.000',
}
export const PERIOD_LABEL: Record<PeriodLevel, string> = {
  daily: 'Diário',
  weekly: 'Semanal',
  monthly: 'Mensal',
  quarterly: 'Trimestral',
}

export function asCase(value: string | undefined | null): CaseKey {
  return value === 'last_candle' ? 'last_candle' : 'last_candles'
}

export function asTf(value: string | undefined | null): TfKey {
  return value === 'm1' ? 'm1' : 'm5'
}

export function asLot(value: string | undefined | null): LotKey {
  return value === 'scaled' ? 'scaled' : 'fixed'
}

export function clock(value: string | null | undefined) {
  if (!value) return '—'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return value.replace('T', ' ').slice(0, 19)
  return d.toLocaleString('pt-BR')
}

export function dayLabel(value: string | null | undefined) {
  if (!value) return '—'
  const [y, m, d] = value.slice(0, 10).split('-')
  if (!y || !m || !d) return value.slice(0, 10)
  return `${d}/${m}/${y}`
}
