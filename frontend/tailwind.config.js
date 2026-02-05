/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    DEFAULT: '#2C3E50',
                    dark: '#1a252f',
                },
                accent: {
                    DEFAULT: '#18BC9C',
                    hover: '#14a387',
                },
                alert: {
                    DEFAULT: '#FF6B6B',
                    hover: '#ff5252',
                },
                dark: {
                    DEFAULT: '#0f172a',
                    lighter: '#1e293b',
                    lightest: '#334155',
                }
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'subtle-float': 'float 5s ease-in-out infinite',
            },
            keyframes: {
                float: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-10px)' },
                }
            }
        },
    },
    plugins: [],
}
