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

export type Winner = {
  name: string
  params: Params
  score: number
  leakage: Leakage
  metrics: Metrics
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

export type StudyFile = {
  generated_at: string
  disclaimer: string
  how_it_works: string[]
  insights: { worked: string[]; failed: string[]; improve: string[] }
  frozen_configs: string[]
  mt5: { ready: boolean; default_enabled: boolean; steps: string[] }
  instrument: { name: string; point_value: number; tick: number }
  winners: { m1: Winner[]; m5: Winner[] }
  m1: {
    leakage: Leakage
    model_test: { test_direction_hit: number; test_mae_close: number; test_rmse_close: number }
    n_configs: number
    leaderboard: { net_pnl: number; n_trades: number; win_rate: number; profit_factor: number }[]
  }
  m5: {
    leakage: Leakage
    model_test: { test_direction_hit: number; test_mae_close: number; test_rmse_close: number }
    n_configs: number
  }
  sanity_m1_extra_file: {
    leakage: Leakage
    metrics: Metrics
    note: string
  } | null
}
