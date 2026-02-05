import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Play, Pause, RotateCcw } from 'lucide-react'

const WaveformPlayer = ({ audioUrl }) => {
    const containerRef = useRef()
    const wavesurferRef = useRef()
    const [isPlaying, setIsPlaying] = useState(false)
    const [duration, setDuration] = useState(0)

    useEffect(() => {
        if (!audioUrl) return

        const wavesurfer = WaveSurfer.create({
            container: containerRef.current,
            waveColor: '#475569',
            progressColor: '#18BC9C',
            cursorColor: '#18BC9C',
            barWidth: 2,
            barRadius: 3,
            responsive: true,
            height: 80,
            normalize: true,
            partialRender: true
        })

        wavesurfer.load(audioUrl)

        wavesurfer.on('ready', () => {
            setDuration(wavesurfer.getDuration())
        })

        wavesurfer.on('play', () => setIsPlaying(true))
        wavesurfer.on('pause', () => setIsPlaying(false))
        wavesurfer.on('finish', () => setIsPlaying(false))

        wavesurferRef.current = wavesurfer

        return () => wavesurfer.destroy()
    }, [audioUrl])

    const togglePlay = () => {
        wavesurferRef.current?.playPause()
    }

    const restart = () => {
        wavesurferRef.current?.stop()
        wavesurferRef.current?.play()
    }

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    return (
        <div className="bg-dark-lightest/30 p-4 rounded-xl border border-white/5 space-y-4">
            <div ref={containerRef} />

            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <button
                        onClick={togglePlay}
                        className="w-10 h-10 flex items-center justify-center bg-accent text-white rounded-full hover:bg-accent-hover transition-colors shadow-lg"
                    >
                        {isPlaying ? <Pause size={18} fill="white" /> : <Play size={18} fill="white" className="ml-1" />}
                    </button>

                    <button
                        onClick={restart}
                        className="w-10 h-10 flex items-center justify-center bg-white/5 text-slate-400 rounded-full hover:bg-white/10 hover:text-white transition-colors"
                    >
                        <RotateCcw size={18} />
                    </button>
                </div>

                <div className="text-sm font-mono text-slate-500">
                    {formatTime(duration)}
                </div>
            </div>
        </div>
    )
}

export default WaveformPlayer
