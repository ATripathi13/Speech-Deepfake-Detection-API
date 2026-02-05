import { Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Landing from './pages/Landing'
import Upload from './pages/Upload'
import Result from './pages/Result'
import History from './pages/History'

function App() {
    return (
        <div className="min-h-screen flex flex-col">
            <Navbar />
            <main className="flex-1">
                <Routes>
                    <Route path="/" element={<Landing />} />
                    <Route path="/upload" element={<Upload />} />
                    <Route path="/result" element={<Result />} />
                    <Route path="/history" element={<History />} />
                    <Route path="*" element={<Navigate to="/" />} />
                </Routes>
            </main>
            <footer className="py-8 px-6 border-t border-white/5 bg-dark">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4 text-slate-500 text-sm">
                    <div>© 2024 VoiceShield AI. Federal Cybersecurity Compliance.</div>
                    <div className="flex gap-6">
                        <a href="#" className="hover:text-white transition-colors">Documentation</a>
                        <a href="#" className="hover:text-white transition-colors">API Keys</a>
                        <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
                    </div>
                </div>
            </footer>
        </div>
    )
}

export default App
