import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// If your inference server runs on another port during development,
// forward `/api/chat` to it instead of dealing with CORS:
//
// export default defineConfig({
//   plugins: [react()],
//   server: {
//     proxy: {
//       '/api': {
//         target: 'http://localhost:8000',
//         changeOrigin: true,
//       },
//     },
//   },
// });

export default defineConfig({
  plugins: [react()],
});