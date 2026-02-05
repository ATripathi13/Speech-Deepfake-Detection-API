import { useState } from 'react'
import { ShieldAlert, ShieldCheck, Search, Filter, Trash2, ExternalLink } from 'lucide-react'
import { useStore } from '../store/useStore'
import { clsx } from 'clsx'

const History = () => {
    const { history, clearHistory } = useStore()
    const [filter, setFilter] = useState('ALL')
    const [search, setSearch] = useState('')

    const filteredHistory = history.filter(item => {
        const matchesFilter = filter === 'ALL' || item.classification === filter
        const matchesSearch = item.fileName.toLowerCase().includes(search.toLowerCase())
        return matchesFilter && matchesSearch
    })

    return (
        <div className="max-w-6xl mx-auto px-6 py-12">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-10">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Detection History</h1>
                    <p className="text-slate-400">Review and audit past forensic analyses</p>
                </div>

                <div className="flex items-center gap-3">
                    <button
                        onClick={clearHistory}
                        className="btn-outline border-alert/20 text-alert hover:bg-alert/10 hover:border-alert/40 px-4 py-2 text-sm"
                    >
                        <Trash2 size={16} />
                        Clear Audit Log
                    </button>
                </div>
            </div>

            <div className="glass-card mb-8">
                <div className="p-4 border-b border-white/5 flex flex-col md:flex-row gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search files..."
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            className="w-full bg-white/5 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:border-accent/50 transition-colors"
                        />
                    </div>

                    <div className="flex items-center gap-2 overflow-x-auto pb-2 md:pb-0">
                        <Filter size={16} className="text-slate-500 mr-2 shrink-0" />
                        {['ALL', 'HUMAN', 'AI_GENERATED'].map(type => (
                            <button
                                key={type}
                                onClick={() => setFilter(type)}
                                className={clsx(
                                    "px-3 py-1.5 rounded-md text-xs font-semibold whitespace-nowrap transition-all",
                                    filter === type
                                        ? "bg-accent text-white shadow-lg shadow-accent/20"
                                        : "bg-white/5 text-slate-400 hover:text-white"
                                )}
                            >
                                {type.replace('_', ' ')}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b border-white/5 bg-white/[0.02]">
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">Classification</th>
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">File Name</th>
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">Confidence</th>
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">Language</th>
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">Timestamp</th>
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {filteredHistory.length > 0 ? filteredHistory.map((item) => (
                                <tr key={item.id} className="hover:bg-white/[0.02] transition-colors group">
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className={clsx(
                                            "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider",
                                            item.classification === 'AI_GENERATED' ? "bg-alert/10 text-alert" : "bg-accent/10 text-accent"
                                        )}>
                                            {item.classification === 'AI_GENERATED' ? <ShieldAlert size={12} /> : <ShieldCheck size={12} />}
                                            {item.classification === 'AI_GENERATED' ? 'AI' : 'Human'}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className="text-sm font-medium text-white group-hover:text-accent transition-colors">{item.fileName}</span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="flex items-center gap-2">
                                            <span className="text-sm font-mono text-slate-300">{(item.confidenceScore * 100).toFixed(0)}%</span>
                                            <div className="h-1 w-12 bg-slate-800 rounded-full overflow-hidden hidden sm:block">
                                                <div
                                                    className={clsx("h-full", item.classification === 'AI_GENERATED' ? "bg-alert" : "bg-accent")}
                                                    style={{ width: `${item.confidenceScore * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className="text-sm text-slate-400">{item.language}</span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className="text-sm text-slate-500">
                                            {new Date(item.timestamp).toLocaleDateString()} {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right">
                                        <button className="p-2 text-slate-500 hover:text-white transition-colors">
                                            <ExternalLink size={16} />
                                        </button>
                                    </td>
                                </tr>
                            )) : (
                                <tr>
                                    <td colSpan="6" className="px-6 py-20 text-center text-slate-500 italic">
                                        No detection logs found. Start an analysis to see results here.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="flex justify-between items-center text-xs text-slate-600">
                <p>Showing {filteredHistory.length} of {history.length} records</p>
                <p>AES-256 Encrypted Storage</p>
            </div>
        </div>
    )
}

export default History
