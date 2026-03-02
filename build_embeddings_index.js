#!/usr/bin/env node

/**
 * Build an embeddings index for NAICS 2017 docs.
 *
 * Input: JSONL records from scrape (out/naics-2017.sharded.jsonl)
 * Output:
 *   - out/naics-2017-embeddings.jsonl  (one line per code: metadata + embedding base64)
 *
 * Requires: OPENAI_API_KEY
 *
 * Usage:
 *   node build_embeddings_index.js \
 *     --in=out/naics-2017.sharded.jsonl \
 *     --out=out/naics-2017-embeddings.jsonl \
 *     --model=text-embedding-3-small
 */

const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const OpenAI = require('openai');

function pickArg(name, def) {
  const p = `--${name}=`;
  const a = process.argv.find((x) => x.startsWith(p));
  return a ? a.slice(p.length) : def;
}

function normalize(s) {
  return String(s || '')
    .replace(/\u00a0/g, ' ')
    .replace(/[\t ]+/g, ' ')
    .trim();
}

function float32ToBase64(arr) {
  const f32 = Float32Array.from(arr);
  return Buffer.from(f32.buffer).toString('base64');
}

const inPath = pickArg('in', path.join(__dirname, 'out', 'naics-2017.sharded.jsonl'));
const outPath = pickArg('out', path.join(__dirname, 'out', 'naics-2017-embeddings.jsonl'));
const model = pickArg('model', process.env.EMBED_MODEL || 'text-embedding-3-small');
const batchSize = Number(pickArg('batch', '64'));

if (!process.env.OPENAI_API_KEY) {
  console.error('Missing OPENAI_API_KEY in environment');
  process.exit(1);
}
if (!fs.existsSync(inPath)) {
  console.error(`Missing input: ${inPath}`);
  process.exit(1);
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

(async () => {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const out = fs.createWriteStream(outPath, { flags: 'w' });

  const rl = readline.createInterface({
    input: fs.createReadStream(inPath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });

  const pending = [];
  let n = 0;

  async function flush() {
    if (!pending.length) return;
    const inputs = pending.map((p) => p.search_text);

    const resp = await client.embeddings.create({
      model,
      input: inputs,
    });

    for (let i = 0; i < pending.length; i++) {
      const p = pending[i];
      const emb = resp.data[i].embedding;
      const line = {
        version: 2017,
        code: p.code,
        title: p.title,
        parents: p.parents,
        source_url: p.source_url,
        // Store only for retrieval/rerank context (keep it compact-ish)
        definition: p.definition,
        illustrative_examples: p.illustrative_examples,
        index_entries: p.index_entries,
        embedding_model: model,
        embedding_b64_f32: float32ToBase64(emb),
      };
      out.write(JSON.stringify(line) + '\n');
      n++;
      if (n % 200 === 0) process.stdout.write(`\r[embed] wrote ${n}`);
    }

    pending.length = 0;
  }

  for await (const line of rl) {
    const l = line.trim();
    if (!l) continue;
    const rec = JSON.parse(l);

    const code = String(rec.code);
    const title = normalize(rec.title);
    const definition = normalize(rec.definition);
    const examples = Array.isArray(rec.illustrative_examples) ? rec.illustrative_examples.map(normalize).filter(Boolean) : [];
    const indexEntries = Array.isArray(rec.index_entries) ? rec.index_entries.map(normalize).filter(Boolean) : [];

    // Build the text we embed.
    // Emphasize title + index entries heavily.
    const searchText = normalize([
      `NAICS ${code}: ${title}`,
      definition ? `Definition: ${definition}` : '',
      indexEntries.length ? `Index entries: ${indexEntries.join('; ')}` : '',
      examples.length ? `Illustrative examples: ${examples.join('; ')}` : ''
    ].filter(Boolean).join('\n'));

    pending.push({
      code,
      title,
      parents: rec.parents || [],
      source_url: rec.source_url || '',
      definition,
      illustrative_examples: examples,
      index_entries: indexEntries,
      search_text: searchText,
    });

    if (pending.length >= batchSize) {
      await flush();
    }
  }

  await flush();
  out.end();
  console.log(`\n[embed] done. wrote ${n} records -> ${outPath}`);
})();
