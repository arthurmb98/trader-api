import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function brl(value: number) {
  return value.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' })
}

export function pct(value: number, digits = 1) {
  return `${value.toFixed(digits)}%`
}

export function num(value: number, digits = 1) {
  return value.toLocaleString('pt-BR', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

export async function readJson<T>(res: Response): Promise<T> {
  const text = await res.text()
  try {
    return JSON.parse(text) as T
  } catch {
    const hint = res.status === 404 ? 'API do paper não está neste host.' : `HTTP ${res.status}`
    throw new Error(hint)
  }
}

const API_ROOTS = ['', 'http://127.0.0.1:8000']

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  let last: unknown
  for (const root of API_ROOTS) {
    try {
      const res = await fetch(`${root}${path}`, { cache: 'no-store', ...init })
      if (res.ok) return res
      last = res
    } catch (err) {
      last = err
    }
  }
  if (last instanceof Response) return last
  throw last instanceof Error ? last : new Error('Failed to fetch')
}
