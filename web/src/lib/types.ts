export type Leakage = {
  n_train: number
  n_test_original: number
  n_removed: number
  n_test_clean: number
  removed_by_key: number
  removed_by_ohlc: number
  train_file: string
  test_file: string
  train_start: string | null
  train_end: string | null
  test_start: string | null
  test_end: string | null
}

export type Metrics = {
  n_candles: number
  n_trades: number
  n_wins: number
  n_losses: number
  win_rate: number
  net_pnl: number
  final_bank: number
  initial_bank: number
  avg_win: number
  avg_loss: number
  median_win: number
  median_loss: number
  avg_points_win: number
  avg_points_loss: number
  profit_factor: number
  max_drawdown: number
  max_drawdown_pct: number
  expectancy: number
  trades_per_candle_pct: number
  hourly: Record<string, { trades: number; wins: number; pnl: number }>
  equity: { t: string; bank: number }[]
  max_contracts?: number
}

export type YearSlice = {
  n_trades: number
  n_wins: number
  win_rate: number
  net_pnl: number
  max_drawdown: number
  max_drawdown_pct: number
}

export type PeriodRow = { t: string; pnl: number; n_trades: number; n_wins: number }

export type PeriodExtremes = {
  best: PeriodRow | null
  worst: PeriodRow | null
  avg: number
  positive_pct: number
}

export type PeriodBreakdown = {
  daily: PeriodRow[]
  weekly: PeriodRow[]
  monthly: PeriodRow[]
  summary: {
    day: PeriodExtremes
    week: PeriodExtremes
    month: PeriodExtremes
    n_days: number
  }
}

export type Params = {
  name: string
  account: {
    initial_bank: number
    contracts: number
    point_value: number
    contract_cost: number
  }
  instrument: { symbol: string; tick_size: number }
  data: { train_csv: string; test_csv: string; timeframe: string }
  risk: Record<string, number | string | boolean>
  filters: Record<string, number | string | boolean | null>
  execution: Record<string, number | string | boolean>
  mt5: Record<string, number | string | boolean>
}

export type RunSide = {
  metrics: Metrics
  by_year?: Record<string, YearSlice>
  by_period?: PeriodBreakdown
  max_contracts?: number
  contracts_path?: { t: string | null; contracts: number; bank: number }[]
}

export type Winner = {
  name: string
  params: Params
  score: number
  leakage: Leakage
  metrics: Metrics
  by_year?: Record<string, YearSlice>
  by_period?: PeriodBreakdown
  lot_fixed?: RunSide
  lot_scaled?: RunSide
  linear?: RunSide
  one_contract?: RunSide
  trades: {
    side: string
    entry_time: string
    exit_time: string
    pnl: number
    result: string
    hour: number
    points: number
  }[]
  model_test?: { test_direction_hit: number; test_mae_close: number }
}

export type StudyBlock = {
  n_configs: number
  n_viable?: number
  leaderboard: { net_pnl: number; n_trades: number; win_rate: number; profit_factor: number }[]
}

export type BankKey = '500' | '1000'
export type CaseKey = 'last_candle' | 'last_candles'
export type TfKey = 'm1' | 'm5'

export type ParecerMonthly = {
  case: string
  bank: number
  name: string
  timeframe: string
  label: string
  avg_fixed: number
  avg_scaled: number
  dd_pct: number
  dd_abs: number
  net_pnl: number
  n_months: number
}

export type ParecerCaseAvg = {
  case: string
  bank: number
  name: string | null
  timeframe?: string
  label: string | null
  avg_fixed: number | null
  avg_scaled: number | null
  n_months: number
}

export type Parecer = {
  headline: string
  ml_hit: { m1: number; m5: number }
  n_months_note: string
  by_case?: ParecerCaseAvg[]
  monthly: ParecerMonthly[]
  strategy: string[]
  dd_floor: string
}

export type StudyFile = {
  generated_at: string
  disclaimer: string
  how_it_works: string[]
  insights: { worked: string[]; failed: string[]; improve: string[] }
  parecer?: Parecer
  frozen_configs: string[]
  mt5: { ready: boolean; default_enabled: boolean; steps: string[] }
  instrument: { name: string; point_value: number; tick: number; contracts?: number }
  banks: number[]
  cases?: CaseKey[]
  case_labels?: Record<string, string>
  lookback?: { m1: number; m5: number }
  timeframes_list?: TfKey[]
  timeframe_labels?: Record<string, string>
  n_configs_total: number
  timeframes: {
    m1: { leakage: Leakage; model_test: { test_direction_hit: number; test_mae_close: number; test_rmse_close: number }; lookback?: number }
    m5: { leakage: Leakage; model_test: { test_direction_hit: number; test_mae_close: number; test_rmse_close: number }; lookback?: number }
  }
  studies: Record<string, Record<string, StudyBlock>>
  winners: Record<string, Record<string, Record<string, Winner[]> | Winner[]>>
}
