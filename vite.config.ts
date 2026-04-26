import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  base: '/Sensora/',
  build: {
    outDir: 'dist',
  },
  plugins: [tailwindcss()],
});
