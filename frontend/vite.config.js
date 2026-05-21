import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     host: true,
//     proxy: {
//       '/health':    'http://localhost:8000',
//       '/documents': 'http://localhost:8000',
//       '/upload':    'http://localhost:8000',
//       '/query':     'http://localhost:8000',
//       '/reset':     'http://localhost:8000',
//       '/stream':    'http://localhost:8000',
//     }
//   }
// })

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      '/health':    'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/upload':    'http://localhost:8000',
      '/query':     'http://localhost:8000',
      '/reset':     'http://localhost:8000',
      '/stream':    'http://localhost:8000',
    }
  }
})