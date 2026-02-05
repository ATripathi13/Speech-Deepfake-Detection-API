import { Link, useLocation } from 'react-router-dom'
import { Shield, History, Upload as UploadIcon, BarChart3 } from 'lucide-react'
import { clsx } from 'clsx'

const Navbar = () => {
    const location = useLocation()

    const navItems = [
        { path: '/upload', label: 'Detect', icon: UploadIcon },
        { path: '/history', label: 'Dashboard', icon: History },
    ]

    return (
        <nav className="sticky top-0 z-50 bg-dark/80 backdrop-blur-lg border-b border-white/5 px-6 py-4">
            <div className="max-w-7xl mx-auto flex items-center justify-between">
                <Link to="/" className="flex items-center gap-2 group">
                    <div className="bg-accent/20 p-2 rounded-lg group-hover:bg-accent/30 transition-colors">
                        <Shield className="w-6 h-6 text-accent" />
                    </div>
                    <span className="text-xl font-bold tracking-tight text-white">VoiceShield <span className="text-accent">AI</span></span>
                </Link>

                <div className="flex items-center gap-1">
                    {navItems.map((item) => (
                        <Link
                            key={item.path}
                            to={item.path}
                            className={clsx(
                                "px-4 py-2 rounded-lg flex items-center gap-2 transition-all font-medium",
                                location.pathname === item.path
                                    ? "bg-white/10 text-white"
                                    : "text-slate-400 hover:text-white hover:bg-white/5"
                            )}
                        >
                            <item.icon className="w-4 h-4" />
                            {item.label}
                        </Link>
                    ))}
                </div>
            </div>
        </nav>
    )
}

export default Navbar
