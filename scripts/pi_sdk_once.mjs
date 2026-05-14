#!/usr/bin/env node
/**
 * Invoke Pi programmatically from Python via one JSON request on stdin.
 *
 * Input JSON:
 * {
 *   "systemPrompt": "...",
 *   "prompt": "...",
 *   "model": "provider/model-id",
 *   "thinking": "off",
 *   "tools": ["read", "bash"]
 * }
 *
 * Output: assistant text on stdout. Diagnostics/errors go to stderr.
 */
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";

function resolvePiPackageRoot() {
  if (process.env.PI_CODING_AGENT_SDK_ROOT) return process.env.PI_CODING_AGENT_SDK_ROOT;
  try {
    const packageJsonUrl = import.meta.resolve("@earendil-works/pi-coding-agent/package.json");
    return path.dirname(new URL(packageJsonUrl).pathname);
  } catch {
    // Fall back to the globally installed `pi` launcher used on this Mac.
  }

  const piBin = execFileSync("which", ["pi"], { encoding: "utf8" }).trim();
  const realPiBin = realpathSync(piBin);
  if (realPiBin.includes("/node_modules/@earendil-works/pi-coding-agent/dist/cli.js")) {
    return path.dirname(path.dirname(realPiBin));
  }
  const launcher = readFileSync(realPiBin, "utf8");
  const match = launcher.match(/node"?\s+"([^"]+\/node_modules\/@earendil-works\/pi-coding-agent\/dist\/cli\.js)"/);
  if (!match) throw new Error(`Could not locate pi SDK package root from launcher: ${piBin} -> ${realPiBin}`);
  const cliPath = match[1].replace(/^\$basedir\//, `${path.dirname(realPiBin)}/`);
  return path.dirname(path.dirname(cliPath));
}

function parseModelPattern(pattern) {
  if (!pattern || typeof pattern !== "string") return undefined;
  const slash = pattern.indexOf("/");
  if (slash <= 0) return undefined;
  return { provider: pattern.slice(0, slash), id: pattern.slice(slash + 1).replace(/:(off|minimal|low|medium|high|xhigh)$/, "") };
}

async function readStdinJson() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  if (!raw) throw new Error("missing JSON request on stdin");
  return JSON.parse(raw);
}

const request = await readStdinJson();
if (!request.dryRun && (!request.prompt || typeof request.prompt !== "string")) {
  throw new Error("request.prompt must be a non-empty string");
}

const totalStart = performance.now();
const packageRoot = resolvePiPackageRoot();
const sdkUrl = pathToFileURL(path.join(packageRoot, "dist", "index.js")).href;
if (!existsSync(new URL(sdkUrl))) throw new Error(`Pi SDK entrypoint not found: ${sdkUrl}`);

const {
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
  SettingsManager,
} = await import(sdkUrl);

const cwd = request.cwd || process.cwd();
const agentDir = process.env.PI_CODING_AGENT_DIR || path.join(process.env.HOME || cwd, ".pi", "agent");
const settingsManager = SettingsManager.inMemory({
  defaultThinkingLevel: request.thinking || "off",
});
const authStorage = AuthStorage.create();
const modelRegistry = ModelRegistry.create(authStorage);
const modelParts = parseModelPattern(request.model || process.env.PI_WORKER_MODEL);
const model = modelParts ? modelRegistry.find(modelParts.provider, modelParts.id) : undefined;

const fullResources = Boolean(request.fullResources);
const resourceLoader = new DefaultResourceLoader({
  cwd,
  agentDir,
  settingsManager,
  noExtensions: !fullResources,
  noSkills: !fullResources,
  noPromptTemplates: !fullResources,
  noThemes: !fullResources,
  noContextFiles: !fullResources,
  appendSystemPrompt: fullResources && request.systemPrompt ? [request.systemPrompt] : [],
  systemPromptOverride: fullResources ? undefined : () => request.systemPrompt || "You are a concise assistant.",
});
await resourceLoader.reload();

const sessionStart = performance.now();
const { session } = await createAgentSession({
  cwd,
  model,
  authStorage,
  modelRegistry,
  resourceLoader,
  thinkingLevel: request.thinking || "off",
  tools: Array.isArray(request.tools) ? request.tools : [],
  sessionManager: SessionManager.inMemory(cwd),
  settingsManager,
});

const startupMs = Math.round((performance.now() - totalStart) * 10) / 10;
const sessionMs = Math.round((performance.now() - sessionStart) * 10) / 10;

if (request.dryRun) {
  session.dispose();
  process.stdout.write(JSON.stringify({ ok: true, startup_ms: startupMs, session_ms: sessionMs }));
  process.exit(0);
}

let text = "";
session.subscribe((event) => {
  if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
    text += event.assistantMessageEvent.delta;
  }
});

try {
  await session.prompt(request.prompt);
  process.stdout.write(text.trim());
} finally {
  session.dispose();
}
