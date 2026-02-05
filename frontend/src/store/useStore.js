import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useStore = create(
    persist(
        (set) => ({
            history: [],
            currentResult: null,

            setResult: (result) => set({ currentResult: result }),

            addHistory: (item) => set((state) => ({
                history: [
                    { ...item, id: crypto.randomUUID(), timestamp: new Date().toISOString() },
                    ...state.history
                ].slice(0, 50) // Keep last 50
            })),

            clearHistory: () => set({ history: [] }),

            clearCurrentResult: () => set({ currentResult: null }),
        }),
        {
            name: 'voice-shield-storage',
        }
    )
)
