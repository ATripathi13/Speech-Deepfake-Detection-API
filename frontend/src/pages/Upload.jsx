import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload as UploadIcon, FileAudio, X, AlertCircle, Loader2, Sparkles } from 'lucide-react'
import { useStore } from '../store/useStore'
import { detectDeepfake } from '../api/client'
import WaveformPlayer from '../components/WaveformPlayer'

const LANGUAGES = [
    { id: 'English', label: 'English' },
    { id: 'Tamil', label: 'Tamil' },
    { id: 'Hindi', label: 'Hindi' },
    { id: 'Telugu', label: 'Telugu' },
    { id: 'Malayalam', label: 'Malayalam' },
]

const Upload = () => {
    const [file, setFile] = useState(null)
    const [previewUrl, setPreviewUrl] = useState(null)
    const [language, setLanguage] = useState('English')
    const [isUploading, setIsUploading] = useState(false)
    const [error, setError] = useState(null)

    const { setResult, addHistory } = useStore()
    const navigate = useNavigate()

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0]
        if (selectedFile) {
            if (selectedFile.size > 10 * 1024 * 1024) {
                setError("File size exceeds 10MB limit")
                return
            }
            setFile(selectedFile)
            setPreviewUrl(URL.createObjectURL(selectedFile))
            setError(null)
        }
    }

    const clearFile = () => {
        setFile(null)
        setPreviewUrl(null)
        setError(null)
    }

    const convertToBase64 = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader()
            reader.readAsDataURL(file)
            reader.onload = () => {
                const base64String = reader.result.split(',')[1]
                resolve(base64String)
            }
            reader.onerror = (error) => reject(error)
        })
    }

    const handleSubmit = async () => {
        if (!file) return

        setIsUploading(true)
        setError(null)

        try {
            const base64 = await convertToBase64(file)
            const result = await detectDeepfake(base64, language, file.name.split('.').pop())

            const enrichedResult = {
                ...result,
                fileName: file.name,
                fileSize: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
            }

            setResult(enrichedResult)
            addHistory(enrichedResult)
            navigate('/result')
        } catch (err) {
            setError(err.detail || "Analysis failed. Please try again.")
        } finally {
            setIsUploading(false)
        }
    }

    return (
        <div className="max-w-3xl mx-auto px-6 py-12">
            <div className="text-center mb-10">
                <h1 className="text-3xl font-bold text-white mb-2">Initialize Detection</h1>
                <p className="text-slate-400">Upload an audio sample for neural forensic analysis</p>
            </div>

            <div className="space-y-8">
                {/* Upload Box */}
                {!file ? (
                    <label className="group relative block w-full aspect-[2/1] border-2 border-dashed border-white/10 rounded-2xl cursor-pointer hover:border-accent/40 hover:bg-white/[0.02] transition-all overflow-hidden">
                        <input type="file" accept="audio/*" className="hidden" onChange={handleFileChange} />
                        <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
                            <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 group-hover:bg-accent/10 transition-all duration-300">
                                <UploadIcon className="w-8 h-8 text-slate-400 group-hover:text-accent" />
                            </div>
                            <p className="text-lg font-medium text-white mb-1">Drop audio file here</p>
                            <p className="text-sm text-slate-500">Supports MP3, WAV, OGG (Max 10MB)</p>
                        </div>
                    </label>
                ) : (
                    <div className="glass-card p-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className="flex items-start justify-between mb-6">
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-12 bg-accent/20 rounded-xl flex items-center justify-center">
                                    <FileAudio className="w-6 h-6 text-accent" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white break-all">{file.name}</h3>
                                    <p className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB • {file.type || 'audio/mpeg'}</p>
                                </div>
                            </div>
                            <button
                                onClick={clearFile}
                                className="p-2 hover:bg-white/5 rounded-lg text-slate-500 hover:text-white transition-colors"
                            >
                                <X size={20} />
                            </button>
                        </div>

                        <WaveformPlayer audioUrl={previewUrl} />

                        <div className="mt-8 pt-8 border-t border-white/5 space-y-6">
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 px-1">Language Context</label>
                                    <select
                                        value={language}
                                        onChange={(e) => setLanguage(e.target.value)}
                                        className="input-field cursor-pointer"
                                    >
                                        {LANGUAGES.map(lang => (
                                            <option key={lang.id} value={lang.id} className="bg-dark">{lang.label}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>

                            {error && (
                                <div className="p-4 bg-alert/10 border border-alert/20 rounded-lg flex gap-3 text-alert text-sm">
                                    <AlertCircle size={18} className="shrink-0" />
                                    {error}
                                </div>
                            )}

                            <button
                                onClick={handleSubmit}
                                disabled={isUploading}
                                className="btn-primary w-full py-4 text-lg shadow-accent/20"
                            >
                                {isUploading ? (
                                    <>
                                        <Loader2 className="w-6 h-6 animate-spin" />
                                        Analyzing Forensic Artifacts...
                                    </>
                                ) : (
                                    <>
                                        <Sparkles className="w-5 h-5 fill-white/20" />
                                        Start AI Detection
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default Upload
