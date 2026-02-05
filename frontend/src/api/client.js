import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const API_KEY = import.meta.env.VITE_API_KEY || 'demo-key-12345'

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY
    }
})

export const detectDeepfake = async (audioBase64, language, format = 'mp3') => {
    try {
        const response = await apiClient.post('/api/voice-detection', {
            language,
            audioFormat: format,
            audioBase64
        })
        return response.data
    } catch (error) {
        console.error('API Error:', error)
        throw error.response?.data || { detail: 'Network error or server unavailable' }
    }
}

export const checkHealth = async () => {
    try {
        const response = await apiClient.get('/health')
        return response.data
    } catch (error) {
        return { status: 'unhealthy' }
    }
}

export default apiClient
