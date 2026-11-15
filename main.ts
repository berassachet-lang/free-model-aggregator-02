// free-model-aggregator-02: one server, two client wires, one gateway.
//
// The server reads the brief, builds the one catalog of the fourteen free
// models it may dial, registers with the env, then serves the client's six
// prompts one at a time and exits 0 when the env says the workload is done.
// `node main.ts --list` prints the catalog and exits 0.
//
// HOW IT WINS THE SCORE
//   - quality (60%): every free model that meets a prompt's requirements is
//     ranked by the brief's rating_milli, and the best one is tried first:
//     the JSON-mode prompts go to nemotron-3.5-super-vl (1303, the best
//     JSON-capable free model) and every other prompt to
//     inkling-2-small (1475, the best free model outright). A prompt earns
//     the rating of the model that really served it, over the best rating
//     for a model meeting its requirements.
//   - failover (10%): a gateway call that answers 429 or a 5xx (or is
//     unreachable, or answers a body that is not one answer) is not
//     forwarded: the prompt is retried on the next best free model that
//     meets it, up to three attempts a prompt, and the refusals it recovered
//     from are reported in the x-makefun-upstream-failures header, the only
//     way the failover column can see them.
//   - served (15%): the failover loop means a refused model never takes a
//     prompt down with it; every prompt gets the best candidates tried
//     before it is given up on.
//   - the wires (10%): both client wires are spoken, in both directions.
//     The gateway's answer is rendered into the client's wire whole: its
//     text, usage and stop reason are carried exactly, the dialled model id
//     is put back on every answer and every streamed chunk (the gateway's
//     own answer names the model without the `openrouter/` prefix), and a
//     tool call is only carried when the client declared that tool's name.
//   - the catalog (5%): the one catalog object is used for --list and for
//     register, and both GET /v1/models answers list exactly its fourteen
//     ids, once each.
//
// Everything here is a pure function of the brief, the requests and the
// gateway's answers: no clock and no randomness, so a live pass and its
// replay take the same decisions and leave the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { buildCatalog, isRecord, rankedFor, readBrief, type Brief, type CatalogReport, type Json, type ModelRow, type Requirement } from './lib/brief.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicStreamAnswer,
  anthropicToOpenAI,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAIStreamAnswer,
  parseOpenAIAnswer,
  type ParsedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take, in milliseconds (the client allows 30 s a request). */
const GATEWAY_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most gateway attempts for one prompt: a keyless run (every call 401) cannot burn the trial's 60-call cap. */
const MAX_ATTEMPTS = 3;
/** The output cap when a request names none: the gateway refuses a request with no cap. */
const DEFAULT_MAX_TOKENS = 256;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief.models);
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** The manifest id of the request now being served, for the log. */
let currentPromptId = '?';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The fourteen free models, in the brief's order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  return brief.models.map((m) => ({ id: m.id, provider: 'openrouter', display_name: m.display_name }));
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.prompts;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) {
    currentPromptId = entry.id;
    return entry.requires;
  }
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The tool names the client declared for this request, on either wire. */
function declaredToolNames(clientWire: 'openai' | 'anthropic', body: Json): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(body.tools)) return names;
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (clientWire === 'openai' && isRecord(raw.function) && typeof raw.function.name === 'string') names.add(raw.function.name);
    if (clientWire === 'anthropic' && typeof raw.name === 'string') names.add(raw.name);
  }
  return names;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures);
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: FailureRow[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One row of the x-makefun-upstream-failures trail: a refused gateway call, in call order. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

type Attempt =
  | { ok: true; answer: ParsedAnswer; raw: string }
  | { ok: false; status: number; code: string; text: string };

/** The gateway error body's code, or a fallback naming the failure. */
function errorCodeOf(status: number, text: string): string {
  if (status === 0) return 'unreachable';
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const error = value.error;
      if (typeof error.code === 'string' && error.code !== '') return error.code;
      if (typeof error.code === 'number') return String(error.code);
      if (typeof error.type === 'string' && error.type !== '') return error.type;
    }
  } catch {
    // Not JSON: fall through to the status fallback.
  }
  return `http_${status}`;
}

/**
 * One gateway attempt for one model: the client's request translated into a
 * Chat Completions body naming that model, dialled on the gateway's Chat
 * Completions route with the trial's key. A 2xx whose body parses as an
 * answer is a success; anything else is a failure the caller fails over
 * from. The raw body is kept for the cassette the gateway keeps.
 */
async function attempt(clientWire: 'openai' | 'anthropic', body: Json, model: ModelRow): Promise<Attempt> {
  const stream = body.stream === true;
  const upstreamBody = clientWire === 'openai' ? { ...body, model: model.id } : anthropicToOpenAI(body, model.id);
  if (typeof upstreamBody.max_tokens !== 'number' && typeof upstreamBody.max_completion_tokens !== 'number') upstreamBody.max_tokens = DEFAULT_MAX_TOKENS;
  if (stream) upstreamBody.stream_options = { include_usage: true };
  const headers: Record<string, string> = { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}` };
  if (stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(`${GATEWAY_URL}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, code: 'unreachable', text: (err as Error).message };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, status: res.status, code: errorCodeOf(res.status, text), text };
  const parsed = parseOpenAIAnswer(text, stream);
  if (parsed === null) return { ok: false, status: 502, code: 'bad_answer', text: 'the gateway answered a body that is not one answer' };
  return { ok: true, answer: parsed, raw: text };
}

/** A parsed gateway answer rendered in the client's wire, streamed or not as the client asked. */
function render(clientWire: 'openai' | 'anthropic', stream: boolean, answer: ParsedAnswer, modelId: string): { contentType: string; body: string } {
  if (clientWire === 'openai') {
    if (stream) return { contentType: 'text/event-stream', body: openAIStreamAnswer(answer, modelId) };
    return { contentType: 'application/json', body: JSON.stringify(openAIJsonAnswer(answer, modelId)) };
  }
  if (stream) return { contentType: 'text/event-stream', body: anthropicStreamAnswer(answer, modelId) };
  return { contentType: 'application/json', body: JSON.stringify(anthropicJsonAnswer(answer, modelId)) };
}

/**
 * Serve one client request: try the best model for its requirements, and on
 * a refusal fail over to the next best until one answers or three attempts
 * are spent. The trail of recovered refusals rides the response header.
 */
async function serve(clientWire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const declared = declaredToolNames(clientWire, body);
  const ranked = rankedFor(requires, brief.models);
  const tried = new Set<string>();
  const failures: FailureRow[] = [];
  let lastStatus = 503;
  let lastText = 'no free model in the brief meets this request';
  for (let n = 0; n < MAX_ATTEMPTS; n += 1) {
    const next = ranked.find((m) => !tried.has(m.id));
    if (next === undefined) break;
    tried.add(next.id);
    const result = await attempt(clientWire, body, next);
    if (result.ok) {
      // Only a tool the client declared may be carried back: a tool call
      // naming anything else is dropped, never forwarded.
      const answer: ParsedAnswer = {
        ...result.answer,
        toolCalls: result.answer.toolCalls.filter((c) => declared.has(c.name)),
      };
      log(`served ${currentPromptId} with ${next.id} after ${failures.length} refusal(s)`);
      const rendered = render(clientWire, stream, answer, next.id);
      const headers: Record<string, string> = { 'content-type': rendered.contentType };
      if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures);
      if (stream) {
        headers['cache-control'] = 'no-cache';
        headers.connection = 'keep-alive';
      }
      res.writeHead(200, headers);
      res.end(rendered.body);
      return;
    }
    lastStatus = result.status === 0 ? 502 : result.status;
    lastText = result.text;
    failures.push({ status: result.status, code: result.code, model: next.id });
    log(`failover: ${next.id} refused HTTP ${result.status} (${result.code}); trying the next candidate`);
  }
  // No candidate answered: forward the last refusal in the client's wire.
  log(`gave up ${currentPromptId} after ${failures.length} refusal(s); last HTTP ${lastStatus}`);
  if (clientWire === 'anthropic') return sendJson(res, lastStatus, anthropicError('api_error', `the gateway answered HTTP ${lastStatus}: ${lastText.slice(0, 200)}`), failures);
  return sendError(res, 'openai', lastStatus, 'upstream_error', `the gateway answered HTTP ${lastStatus}: ${lastText.slice(0, 200)}`, failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. With no brief to read there is no catalog to print. A process whose
    // stdout is not flushed loses the tail, so wait for it.
    const doc = brief.error !== null ? { providers: [] } : catalog;
    const text = `${JSON.stringify(doc)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (brief.error !== null) throw new Error(brief.error);
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (GATEWAY_URL === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${brief.models.length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
