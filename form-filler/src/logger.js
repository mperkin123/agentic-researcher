import fs from "node:fs";
import path from "node:path";

export function buildRunLogger({ runDir }) {
  fs.mkdirSync(runDir, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const file = path.join(runDir, `${ts}.jsonl`);
  const stream = fs.createWriteStream(file, { flags: "a" });

  return {
    file,
    async log(obj) {
      stream.write(JSON.stringify(obj) + "\n");
    },
    async close() {
      await new Promise((resolve) => stream.end(resolve));
    },
  };
}
