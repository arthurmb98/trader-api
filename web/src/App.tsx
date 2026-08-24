import { AoVivoPage } from '@/pages/AoVivoPage'
import { LivePage } from '@/pages/LivePage'
import { StudyPage } from '@/pages/StudyPage'

export default function App() {
  const path = window.location.pathname.replace(/\/$/, '') || '/'
  if (path === '/ao-vivo') return <AoVivoPage />
  if (path === '/replay' || path === '/live') return <LivePage />
  return <StudyPage />
}
