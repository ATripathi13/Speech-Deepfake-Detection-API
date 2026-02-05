import { Link } from 'react-router-dom'
import { Shield, Lock, Zap, Search, ChevronRight } from 'lucide-react'

const Landing = () => {
    return (
        <div className="relative isolate overflow-hidden">
            {/* Background Decor */}
            <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
                <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#18BC9C] to-[#2C3E50] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"></div>
            </div>

            <div className="max-w-7xl mx-auto px-6 pt-20 pb-24 sm:pt-32 lg:flex lg:items-center lg:gap-x-10 lg:px-8 lg:pt-40">
                <div className="mx-auto max-w-2xl lg:mx-0 lg:flex-auto">
                    <div className="flex">
                        <div className="relative flex items-center gap-x-4 rounded-full px-4 py-1 text-sm leading-6 text-slate-400 ring-1 ring-white/10 hover:ring-white/20">
                            <span className="font-semibold text-accent">New Release v1.2</span>
                            <span className="h-4 w-px bg-white/10"></span>
                            <a href="#" className="flex items-center gap-x-1">
                                Forensic updates loaded
                                <ChevronRight className="h-3 w-3" />
                            </a>
                        </div>
                    </div>
                    <h1 className="mt-10 max-w-lg text-4xl font-bold tracking-tight text-white sm:text-6xl">
                        Speech Deepfake <span className="text-accent underline decoration-accent/30 underline-offset-8">Detection</span>
                    </h1>
                    <p className="mt-6 text-lg leading-8 text-slate-400">
                        Secure your audio infrastructure with real-time neural analysis. Identify AI-generated voices, deepfakes, and synthetic artifacts with forensic precision.
                    </p>
                    <div className="mt-10 flex items-center gap-x-6">
                        <Link to="/upload" className="btn-primary px-8 py-4 text-lg">
                            Analyze Audio Now
                        </Link>
                        <a href="#" className="text-sm font-semibold leading-6 text-white flex items-center gap-2 hover:text-accent transition-colors">
                            How it works <span aria-hidden="true">→</span>
                        </a>
                    </div>
                </div>

                <div className="mt-16 sm:mt-24 lg:mt-0 lg:flex-shrink-0 lg:flex-grow">
                    <div className="glass-card p-1">
                        <div className="bg-dark p-6 rounded-xl">
                            <div className="grid grid-cols-2 gap-4">
                                <FeatureCard
                                    icon={Shield}
                                    title="Forensic Analysis"
                                    desc="Detects microscopic spectral anomalies"
                                />
                                <FeatureCard
                                    icon={Lock}
                                    title="Enterprise Auth"
                                    desc="Secure API key protected endpoints"
                                />
                                <FeatureCard
                                    icon={Zap}
                                    title="Real-time"
                                    desc="Results in under 2 seconds"
                                />
                                <FeatureCard
                                    icon={Search}
                                    title="Explainable AI"
                                    desc="Detailed alerts on artifact types"
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

const FeatureCard = ({ icon: Icon, title, desc }) => (
    <div className="p-4 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
        <Icon className="w-6 h-6 text-accent mb-3" />
        <h3 className="font-semibold text-white mb-1">{title}</h3>
        <p className="text-xs text-slate-500">{desc}</p>
    </div>
)

export default Landing
