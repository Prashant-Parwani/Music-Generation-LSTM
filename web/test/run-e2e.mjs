// Bundles test/e2e.jsx (with Tone.js stubbed) and runs it in Node. Usage: npm run test:e2e
import { build } from "esbuild";
import { spawnSync } from "node:child_process";

await build({
  entryPoints: ["test/e2e.jsx"], bundle: true, platform: "node", format: "cjs", jsx: "automatic",
  alias: { tone: "./test/tone-stub.js" }, external: ["jsdom"],
  define: { "import.meta.env": '{"VITE_API_URL":""}' },
  outfile: "test/e2e.bundle.cjs", logLevel: "error", loader: { ".js": "jsx" },
});
const r = spawnSync(process.execPath, ["test/e2e.bundle.cjs"], { stdio: "inherit" });
process.exit(r.status ?? 1);
