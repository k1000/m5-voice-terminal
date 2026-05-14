#!/usr/bin/env node
/**
 * Smoke test for starting Pi programmatically via the SDK.
 *
 * This deliberately measures SDK session construction, not an LLM call, so it is
 * fast, deterministic, and safe to run from `make test`. Use --prompt to also
 * perform a real model request once the session has started.
 */
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";

const args = new Set(process.argv.slice(2));
const thresholdMs = Number(process.env.PI_STARTUP_THRESHOLD_MS || "5000");
const cwd = process.cwd();
const agentDir = process.env.PI_CODING_AGENT_DIR || path.join(process.env.HOME || cwd, ".pi", "agent");
const shouldPrompt = args.has("--prompt");

function resolvePiPackageRoot() {
  if (process.env.PI_CODING_AGENT_SDK_ROOT) {
    return process.env.PI_CODING_AGENT_SDK_ROOT;
  }

  try {
    // Works when the repo has @earendil-works/pi-coding-agent installed locally.
    const packageJsonUrl = import.meta.resolve("@earendil-works/pi-coding-agent/package.json");
    return path.dirname(new URL(packageJsonUrl).pathname);
  } catch {
    // Fall through to the globally installed `pi` launcher.
  }

  const piBin = execFileSync("which", ["pi"], { encoding: "utf8" }).trim();
  const realPiBin = realpathSync(piBin);
  if (realPiBin.includes("/node_modules/@earendil-works/pi-coding-agent/dist/cli.js")) {
    return path.dirname(path.dirname(realPiBin));
  }
  const launcher = readFileSync(realPiBin, "utf8");
  const match = launcher.match(/node"?\s+"([^"]+\/node_modules\/@earendil-works\/pi-coding-agent\/dist\/cli\.js)"/);
  if (!match) {
    throw new Error(`Could not locate pi SDK package root from launcher: ${piBin} -> ${realPiBin}`);
  }
  const launcherDir = path.dirname(realPiBin);
  const cliPath = match[1].replace(/^\$basedir\//, `${launcherDir}/`);
  return path.dirname(path.dirname(cliPath));
}

function msSince(start) {
  return Math.round((performance.now() - start) * 10) / 10;
}

const totalStart = performance.now();
const packageRoot = resolvePiPackageRoot();
const sdkUrl = pathToFileURL(path.join(packageRoot, "dist", "index.js")).href;

if (!existsSync(new URL(sdkUrl))) {
  throw new Error(`Pi SDK entrypoint not found: ${sdkUrl}`);
}

const importStart = performance.now();
const {
  createAgentSession,
  DefaultResourceLoader,
  SessionManager,
  SettingsManager,
} = await import(sdkUrl);
const importMs = msSince(importStart);

const loaderStart = performance.now();
const resourceLoader = new DefaultResourceLoader({
  cwd,
  agentDir,
  settingsManager: SettingsManager.inMemory(),
  noExtensions: true,
  noSkills: true,
  noPromptTemplates: true,
  noThemes: true,
  noContextFiles: true,
  systemPromptOverride: () => "You are a concise voice-terminal assistant.",
  extensionFactories: [],
  skillsOverride: () => ({ skills: [], diagnostics: [] }),
  agentsFilesOverride: () => ({ agentsFiles: [] }),
  promptsOverride: () => ({ prompts: [], diagnostics: [] }),
});
await resourceLoader.reload();
const loaderMs = msSince(loaderStart);

const sessionStart = performance.now();
const { session } = await createAgentSession({
  cwd,
  resourceLoader,
  tools: [],
  sessionManager: SessionManager.inMemory(cwd),
  settingsManager: SettingsManager.inMemory(),
});
const sessionMs = msSince(sessionStart);

let promptMs = 0;
if (shouldPrompt) {
  const promptStart = performance.now();
  session.subscribe((event) => {
    if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
      process.stdout.write(event.assistantMessageEvent.delta);
    }
  });
  await session.prompt("Reply with exactly: pi sdk ok");
  promptMs = msSince(promptStart);
  process.stdout.write("\n");
}

session.dispose();

const totalMs = msSince(totalStart);
const result = {
  ok: totalMs <= thresholdMs,
  total_ms: totalMs,
  threshold_ms: thresholdMs,
  import_ms: importMs,
  resource_loader_ms: loaderMs,
  session_ms: sessionMs,
  prompt_ms: promptMs,
  package_root: packageRoot,
};
console.log(JSON.stringify(result, null, 2));

if (!result.ok) {
  console.error(`Pi SDK startup exceeded ${thresholdMs} ms`);
  process.exit(1);
}
