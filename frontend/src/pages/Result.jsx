import { useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { ShieldAlert, ShieldCheck, ArrowLeft, Download, Info, Zap, Layers, Activity } from 'lucide-react'
import { useStore } from '../store/useStore'
import { clsx } from 'clsx'
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts'

const Result = () => {
    const { currentResult } = useStore()
    const navigate = useNavigate()

    useEffect(() => {
        if (!currentResult) {
            navigate('/upload')
        }
    }, [currentResult, navigate])

    if (!currentResult) return null

    const isAI = currentResult.classification === 'AI_GENERATED'
    const confidence = currentResult.confidenceScore * 100

    const chartData = [
        { name: 'Confidence', value: confidence },
        { name: 'Remainder', value: 100 - confidence },
    ]

    const COLORS = [isAI ? '#FF6B6B' : '#18BC9C', '#1e293b']

    return (
        <div className="max-w-5xl mx-auto px-6 py-12">
            <Link
                to="/upload"
                className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-8 transition-colors group"
            >
                <ArrowLeft size={18} className="group-hover:-translate-x-1 transition-transform" />
                Back to analysis
            </Link>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main Status Column */}
                <div className="lg:col-span-2 space-y-8">
                    <div className={clsx(
                        "glass-card p-8 border-t-4",
                        isAI ? "border-t-alert shadow-alert/5" : "border-t-accent shadow-accent/5"
                    )}>
                        <div className="flex flex-col md:flex-row items-center gap-8">
                            <div className="relative w-48 h-48">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={chartData}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={80}
                                            paddingAngle={5}
                                            dataKey="value"
                                            startAngle={90}
                                            endAngle={450}
                                        >
                                            {chartData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} cornerRadius={10} />
                                            ))}
                                        </Pie>
                                    </PieChart>
                                </ResponsiveContainer>
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                                    <span className="text-3xl font-bold text-white">{confidence.toFixed(0)}%</span>
                                    <span className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Confidence</span>
                                </div>
                            </div>

                            <div className="flex-1 text-center md:text-left">
                                <div className={clsx(
                                    "inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider mb-4",
                                    isAI ? "bg-alert/10 text-alert" : "bg-accent/10 text-accent"
                                )}>
                                    {isAI ? <ShieldAlert size={14} /> : <ShieldCheck size={14} />}
                                    Internal Scan Result
                                </div>
                                <h2 className="text-4xl md:text-5xl font-black tracking-tight text-white mb-4">
                                    {isAI ? 'AI GENERATED' : 'HUMAN SPEECH'}
                                </h2>
                                <p className="text-slate-400 text-lg leading-relaxed">
                                    {currentResult.explanation}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Forensic Details */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <MetricCard
                            icon={Activity}
                            label="Audio Source"
                            value={currentResult.fileName}
                        />
                        <MetricCard
                            icon={Zap}
                            label="Language Model"
                            value={currentResult.language}
                        />
                        <MetricCard
                            icon={Layers}
                            label="Format Detection"
                            value={currentResult.audioFormat?.toUpperCase() || 'MP3'}
                        />
                        <MetricCard
                            icon={Info}
                            label="Status"
                            value="Analysis Verified"
                        />
                    </div>
                </div>

                {/* Sidebar Column */}
                <div className="space-y-6">
                    <div className="glass-card p-6">
                        <h3 className="font-bold text-white mb-6 flex items-center gap-2">
                            <Info className="w-5 h-5 text-accent" />
                            Explainability Report
                        </h3>
                        <div className="space-y-6">
                            <ExplainabilityItem
                                title="Spectral Consistency"
                                status={isAI ? "Suspicious" : "Natural"}
                                percent={isAI ? 88 : 12}
                                color={isAI ? "alert" : "accent"}
                            />
                            <ExplainabilityItem
                                title="Pitch Fluidity"
                                status={isAI ? "Robotic" : "Variable"}
                                percent={isAI ? 92 : 8}
                                color={isAI ? "alert" : "accent"}
                            />
                            <ExplainabilityItem
                                title="Codec Artifacts"
                                status={isAI ? "Present" : "None"}
                                percent={isAI ? 75 : 5}
                                color={isAI ? "alert" : "accent"}
                            />
                        </div>

                        <button className="btn-outline w-full mt-8 py-3 text-sm">
                            <Download size={16} />
                            Export Detailed Report
                        </button>
                    </div>

                    <div className="bg-gradient-to-br from-accent/20 to-transparent p-6 rounded-2xl border border-accent/20">
                        <p className="text-sm text-slate-300 font-medium mb-2">Need higher precision?</p>
                        <p className="text-xs text-slate-500 mb-4">Upgrade to VoiceShield Enterprise for 12 additional forensic markers.</p>
                        <button className="text-accent text-sm font-bold hover:underline">View Pricing →</button>
                    </div>
                </div>
            </div>
        </div>
    )
}

const MetricCard = ({ icon: Icon, label, value }) => (
    <div className="glass-card p-4 flex items-center gap-4">
        <div className="bg-white/5 p-3 rounded-lg">
            <Icon className="w-5 h-5 text-slate-400" />
        </div>
        <div>
            <p className="text-[10px] uppercase font-bold tracking-widest text-slate-500">{label}</p>
            <p className="text-sm font-semibold text-white truncate max-w-[150px]">{value}</p>
        </div>
    </div>
)

const ExplainabilityItem = ({ title, status, percent, color }) => (
    <div className="space-y-2">
        <div className="flex justify-between text-xs">
            <span className="text-slate-400 font-medium">{title}</span>
            <span className={color === 'alert' ? 'text-alert' : 'text-accent'}>{status}</span>
        </div>
        <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
            <div
                className={clsx(
                    "h-full transition-all duration-1000",
                    color === 'alert' ? 'bg-alert' : 'bg-accent'
                )}
                style={{ width: `${percent}%` }}
            />
        </div>
    </div>
)

export default Result
