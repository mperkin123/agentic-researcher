#!/usr/bin/env node

/**
 * NAICS classifier API (hybrid vector search + LLM rerank)
 *
 * Build embeddings file first:
 *   OPENAI_API_KEY=... node build_embeddings_index.js
 *
 * Run:
 *   OPENAI_API_KEY=... node server_vector.js --port=8787 --emb=out/naics-2017-embeddings.jsonl
 */

const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');

const express = require('express');
const { z } = require('zod');
const OpenAI = require('openai');

function pickArg(name, def) {
  const p = `--${name}=`;
  const a = process.argv.find((x) => x.startsWith(p));
  return a ? a.slice(p.length) : def;
}

function b64ToFloat32(b64) {
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

function cosineSim(a, b, normA, normB) {
  return dot(a, b) / (normA * normB + 1e-12);
}

function truncateForEmbedding(text, maxChars = 8000) {
  if (text.length <= maxChars) return text;
  // Keep the beginning and end (often contains key differentiators)
  const head = text.slice(0, Math.floor(maxChars * 0.7));
  const tail = text.slice(-Math.floor(maxChars * 0.3));
  return head + '\n...\n' + tail;
}

const port = Number(pickArg('port', process.env.PORT || '8787'));
const embPath = pickArg('emb', path.join(__dirname, 'out', 'naics-2017-embeddings.jsonl'));
const embedModel = pickArg('embed-model', process.env.EMBED_MODEL || 'text-embedding-3-small');
const llmModel = pickArg('llm-model', process.env.LLM_MODEL || 'gpt-4o-mini');
const topK = Number(pickArg('topk', process.env.TOPK || '25'));

if (!process.env.OPENAI_API_KEY) {
  console.error('Missing OPENAI_API_KEY');
  process.exit(1);
}
if (!fs.existsSync(embPath)) {
  console.error(`Missing embeddings file: ${embPath}`);
  console.error('Build it: OPENAI_API_KEY=... node build_embeddings_index.js');
  process.exit(1);
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function loadEmbeddings() {
  console.log(`[api] loading embeddings from ${embPath}`);
  const rows = [];
  const rl = readline.createInterface({
    input: fs.createReadStream(embPath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });

  let n = 0;
  for await (const line of rl) {
    const l = line.trim();
    if (!l) continue;
    const rec = JSON.parse(l);
    const vec = b64ToFloat32(rec.embedding_b64_f32);
    const nrm = norm(vec);
    rows.push({
      code: rec.code,
      title: rec.title,
      parents: rec.parents || [],
      source_url: rec.source_url || '',
      definition: rec.definition || '',
      illustrative_examples: rec.illustrative_examples || [],
      index_entries: rec.index_entries || [],
      vec,
      nrm,
      embedding_model: rec.embedding_model
    });
    n++;
    if (n % 500 === 0) process.stdout.write(`\r[api] loaded ${n}`);
  }
  console.log(`\n[api] loaded ${rows.length} vectors`);
  return rows;
}

function formatCandidateForLLM(c) {
  // Keep context compact but informative
  const idx = (c.index_entries || []).slice(0, 40);
  const ex = (c.illustrative_examples || []).slice(0, 10);

  return {
    code: c.code,
    title: c.title,
    definition: c.definition,
    index_entries_sample: idx,
    illustrative_examples_sample: ex,
    source_url: c.source_url
  };
}

(async () => {
  const rows = await loadEmbeddings();

  const app = express();
  app.use(express.json({ limit: '20kb' }));

  // Simple local UI
  app.use(express.static(path.join(__dirname, 'public')));

  app.get('/health', (req, res) => {
    res.json({ ok: true, version: 2017, vectors: rows.length, embedModel, llmModel });
  });

  const classifySchema = z.object({
    text: z.string().min(1).max(10_000)
  });

  app.post('/classify', async (req, res) => {
    const parsed = classifySchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: 'invalid_request', details: parsed.error.issues });
    }

    const inputText = parsed.data.text;
    const embedText = truncateForEmbedding(inputText, 8000);

    // 1) Embed query
    const embResp = await client.embeddings.create({
      model: embedModel,
      input: embedText
    });
    const q = Float32Array.from(embResp.data[0].embedding);
    const qn = norm(q);

    // 2) Vector search (linear scan; fine for 1.3k docs)
    const scored = rows.map((r) => ({
      r,
      sim: cosineSim(q, r.vec, qn, r.nrm)
    }));
    scored.sort((a, b) => b.sim - a.sim);

    const shortlist = scored.slice(0, topK);

    // 3) LLM rerank to top 3 w/ 0-100 scores
    const candidates = shortlist.map((x) => formatCandidateForLLM(x.r));

    const system = `You are a NAICS (2017) classification assistant.\n\nGiven a business description and a list of candidate NAICS codes (each with title and sample index entries), select the 3 most likely NAICS codes.\n\nReturn strict JSON only.`;

    const user = {
      business_description: inputText,
      candidates,
      instructions: {
        output: {
          top: [
            { code: 'string', title: 'string', score: 'integer 0-100' }
          ]
        },
        rules: [
          'Return exactly 3 results if possible.',
          'Scores should reflect confidence relative to this input (100 = best fit).',
          'Do not invent codes not in candidates.',
          'Prefer 6-digit codes when the description is specific enough.'
        ]
      }
    };

    const chat = await client.chat.completions.create({
      model: llmModel,
      temperature: 0,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: JSON.stringify(user) }
      ]
    });

    let parsedOut;
    try {
      parsedOut = JSON.parse(chat.choices[0].message.content);
    } catch {
      parsedOut = { top: [] };
    }

    // Attach URLs from our metadata
    const byCode = new Map(candidates.map((c) => [String(c.code), c]));
    const top = Array.isArray(parsedOut.top) ? parsedOut.top : [];
    const enriched = top.slice(0, 3).map((t) => {
      const c = byCode.get(String(t.code));
      return {
        code: String(t.code),
        title: t.title || c?.title || '',
        score: Math.max(0, Math.min(100, Math.round(Number(t.score) || 0))),
        source_url: c?.source_url || ''
      };
    });

    res.json({
      top: enriched,
      debug: {
        top_vector: shortlist.slice(0, 3).map((x) => ({ code: x.r.code, title: x.r.title, similarity: x.sim }))
      }
    });
  });

  app.listen(port, () => {
    console.log(`[naics-vector-api] listening on http://localhost:${port}`);
  });
})();
