// The trial brief: the free-model list and the six prompts' manifest.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) carries two JSON
// documents, and the prose around them does not matter: this file finds each
// by its `schema` member.
//
//   - the free-model list (`free-model-aggregator-02/models/v1`): one row a
//     free model the gateway routes, with the id a request names, its
//     published name, the house's quality rating in milli-points, its context
//     length and what it can do (tools, vision, JSON mode).
//   - the prompts manifest (`free-model-aggregator-02/prompts/v1`): the six
//     prompts the env sends, in order, each with its wire, whether it
//     streams, its weight and the requirements a model must meet to serve it.
//
// Everything here is deterministic: no clock and no randomness, so
// `node main.ts --list` and the register body build the same document.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id a request names. */
export interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface ModelsFile {
  schema: string;
  models: ModelRow[];
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface PromptSpec {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface PromptsFile {
  schema: string;
  prompts: PromptSpec[];
}

/** The brief as the entry reads it. `error` is set when it could not be read at all. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptSpec[];
  error: string | null;
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function boolOr(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function parseModel(raw: unknown): ModelRow | null {
  if (!isRecord(raw) || typeof raw.id !== 'string' || raw.id === '') return null;
  return {
    id: raw.id,
    provider: typeof raw.provider === 'string' ? raw.provider : 'openrouter',
    display_name: typeof raw.display_name === 'string' ? raw.display_name : raw.id,
    rating_milli: numberOr(raw.rating_milli, 0),
    context_length: typeof raw.context_length === 'number' ? raw.context_length : null,
    tools: boolOr(raw.tools, false),
    vision: boolOr(raw.vision, false),
    json: boolOr(raw.json, false),
    free: boolOr(raw.free, true),
  };
}

function parsePrompt(raw: unknown): PromptSpec | null {
  if (!isRecord(raw) || typeof raw.id !== 'string' || raw.id === '') return null;
  const api = raw.api === 'anthropic' ? 'anthropic' : 'openai';
  const requires = isRecord(raw.requires) ? raw.requires : {};
  return {
    id: raw.id,
    api,
    stream: raw.stream === true,
    weight: raw.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
    },
  };
}

/**
 * Read the brief from the file at `path`. The file is prose with two JSON
 * documents in it; each document is found by its `schema` member, one line
 * holding the whole document.
 */
export function readBrief(path: string): Brief {
  const brief: Brief = { models: [], prompts: [], error: null };
  if (path === '') {
    brief.error = 'MAKEFUN_PROMPT_FILE is not set: no brief to read';
    return brief;
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    brief.error = `cannot read the brief at ${path}: ${(err as Error).message}`;
    return brief;
  }
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{"schema":')) continue;
    let doc: unknown;
    try {
      doc = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (!isRecord(doc) || typeof doc.schema !== 'string') continue;
    if (doc.schema === MODELS_SCHEMA && Array.isArray(doc.models)) {
      for (const raw of doc.models) {
        const row = parseModel(raw);
        if (row !== null && row.free) brief.models.push(row);
      }
    } else if (doc.schema === PROMPTS_SCHEMA && Array.isArray(doc.prompts)) {
      for (const raw of doc.prompts) {
        const spec = parsePrompt(raw);
        if (spec !== null) brief.prompts.push(spec);
      }
    }
  }
  if (brief.models.length === 0) brief.error = `the brief at ${path} names no free models`;
  return brief;
}

/** Whether a model meets a prompt's requirements. */
export function meetsRequirements(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools && !model.tools) return false;
  if (requires.vision && !model.vision) return false;
  if (requires.json && !model.json) return false;
  return true;
}

/** The models for one prompt's requirements, best rating first (ties: model id). */
export function rankedFor(requires: Requirement, models: readonly ModelRow[]): ModelRow[] {
  return models
    .filter((m) => meetsRequirements(requires, m))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** The one catalog document: the single openrouter provider, the brief's free models. */
export function buildCatalog(models: readonly ModelRow[]): CatalogReport {
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models: models.map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
      },
    ],
  };
}
