import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, "..");
const projectRoot = resolve(desktopDir, "../..");
const outputRoot = join(desktopDir, "bundled-backend");
const backendOut = join(outputRoot, "backend");
const pythonOut = join(outputRoot, "python");
const sitePackagesOut = join(pythonOut, "site-packages");
const pythonRuntimeOut = join(outputRoot, "python-runtime");

// Node 24's `rmSync({ recursive, force })` silently throws EPERM on
// Windows when the path contains non-ASCII characters and the
// underlying call uses the `\\?\` long-path prefix. Delegate to the
// platform's `rm -rf` equivalent — `rmdir /s /q` on Windows, `rm -rf`
// on Unix — so the wipe is byte-transparent.
function rmSyncOutput(target) {
  if (!existsSync(target)) {
    return;
  }
  if (process.platform === "win32") {
    execFileSync("cmd", ["/c", "rmdir", "/s", "/q", target], { stdio: "ignore" });
  } else {
    execFileSync("rm", ["-rf", target], { stdio: "ignore" });
  }
}

// Node.js ≥ 22's `cpSync({ recursive: true })` silently dies with exit 127
// on Windows when the source or destination path contains non-ASCII
// characters (e.g. `e:\代码运行\券商汇报\...`). Walk the tree manually
// using per-file `cpSync` (no recursive flag) so the build works on
// both Node 22 LTS and Node 24 current. The optional `filter` callback
// mirrors the fs.cpSync filter contract: it receives the source path of
// every entry; return false to skip the entry.
function cpDirRecursive(src, dst, filter) {
  // Mirror fs.cpSync: if src is a file, copy it directly.
  if (!existsSync(src)) {
    throw new Error(`cpDirRecursive: source does not exist: ${src}`);
  }
  const srcStat = statSync(src);
  if (srcStat.isFile()) {
    cpSync(src, dst);
    return;
  }
  if (!srcStat.isDirectory()) {
    throw new Error(`cpDirRecursive: unsupported source type for ${src}`);
  }
  mkdirSync(dst, { recursive: true });
  for (const entry of readdirSync(src, { withFileTypes: true })) {
    const srcPath = join(src, entry.name);
    if (filter && !filter(srcPath)) {
      continue;
    }
    const dstPath = join(dst, entry.name);
    if (entry.isDirectory()) {
      cpDirRecursive(srcPath, dstPath, filter);
    } else if (entry.isSymbolicLink()) {
      const target = readFileSync(srcPath);
      writeFileSync(dstPath, target);
    } else if (entry.isFile()) {
      cpSync(srcPath, dstPath);
    }
  }
}

const backendFiles = [
  "agent_state.py",
  "color_style.py",
  "cross_checker.py",
  "backtest.py",
  "decision_agent.py",
  "default_config.py",
  "graph_setup.py",
  "graph_util.py",
  "history_store.py",
  "indicator_agent.py",
  "pattern_agent.py",
  "proxy_fix.py",
  "report_export.py",
  "requirements.txt",
  "static_util.py",
  "trading_graph.py",
  "trend_agent.py",
  "web_interface.py",
];

// Variant comes from the desktop project's package.json name (the
// tauri-academic / tauri-trader subdirs). It's also accepted as
// --variant=academic|trader on the command line for direct invocation.
// When neither is set, default to "trader" so a misconfigured build
// still produces a runnable trader shell.
const variantArg = process.argv.find((arg) => arg.startsWith("--variant="));
const variant = (
  variantArg
    ? variantArg.slice("--variant=".length)
    : (process.env.QUANTAGENT_APP_VARIANT || "trader")
).toLowerCase();
if (variant !== "academic" && variant !== "trader") {
  throw new Error(
    `Unknown variant: ${variant}. Expected --variant=academic or --variant=trader.`
  );
}

const backendDirs = ["assets", "data_providers", "static", "templates"];

const sitePackages = findSitePackages();
const venvPython = findVenvPython();
const pythonVersion = execFileSync(
  venvPython,
  ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
  { encoding: "utf8" },
).trim();
// Find the base Python install by parsing pyvenv.cfg. sys.base_prefix
// in some Python builds returns a path encoded with a different Windows
// codepage than the actual on-disk directory (Node then can't readdir
// it). pyvenv.cfg stores `home = <real path>` using whatever bytes were
// on disk when the venv was created, so it's authoritative.
const basePrefix = findVenvBasePrefix();

// Node 24's `rmSync({ recursive, force })` silently throws EPERM on
// Windows when the path contains non-ASCII characters (e.g.
// `e:\代码运行\...`) and the underlying call uses the `\\?\` long-path
// prefix. Delegate to a shell `rm -rf` instead, which honours the
// system codepage and is what most build scripts do anyway.
rmSyncOutput(outputRoot);
mkdirSync(backendOut, { recursive: true });
mkdirSync(sitePackagesOut, { recursive: true });
writeFileSync(join(pythonOut, "version.txt"), `${pythonVersion}\n`);

for (const file of backendFiles) {
  copyRequired(join(projectRoot, file), join(backendOut, file));
}

for (const dir of backendDirs) {
  copyRequired(join(projectRoot, dir), join(backendOut, dir));
}



// Generate build-time built-in API keys as an ordinary .py module. The desktop
// app imports this at startup so customers get keys baked in without entering
// them. We generate it here (rather than in a separate CI step) because
// tauri.mjs re-runs prepare-backend during `npm run build`, so any file added
// after prepare would be wiped. Values come from CI environment variables
// (GitHub Secrets); missing values produce empty strings so local/dev builds
// still succeed and rely on a local .env instead.
writeBuiltinKeys(backendOut);

// Bake the build variant into the backend as a generated module. The
// runtime still reads QUANTAGENT_APP_VARIANT (set by lib.rs), but having
// a static fallback means a missing env var still picks the right
// template + features instead of silently defaulting to trader.
writeVariantFile(backendOut, variant);

cpDirRecursive(sitePackages, sitePackagesOut, (source) => {
  const normalized = source.replaceAll("\\", "/");
  return !normalized.includes("/__pycache__/")
    && !normalized.endsWith("/__pycache__")
    && !normalized.endsWith(".pyc")
    && !normalized.includes("/pip/_vendor/cachecontrol/caches/");
});

if (process.platform === "win32") {
  prepareWindowsPythonRuntime(basePrefix);
}

console.log(`Bundled backend: ${backendOut}`);
console.log(`Bundled Python site-packages: ${sitePackagesOut}`);
console.log(`Required Python version: ${pythonVersion}`);

function copyRequired(source, target) {
  if (!existsSync(source)) {
    throw new Error(`Required backend path missing: ${source}`);
  }

  cpDirRecursive(source, target);
}

function writeBuiltinKeys(targetDir) {
  // Read credentials from the environment (set from GitHub Secrets in CI).
  // Defaults keep the generated module valid for local builds without secrets.
  const arkKey = process.env.ARK_API_KEY || "";
  const arkBase = process.env.ARK_BASE_URL || "https://ark.cn-beijing.volces.com/api/coding";
  const arkModel = process.env.ARK_MODEL || "MiniMax-M3";
  const deepseekKey = process.env.DEEPSEEK_API_KEY || "";

  const py = [
    '"""Build-time generated built-in API keys for the desktop app. Do not edit; regenerated by prepare-backend.mjs."""',
    "import os",
    `os.environ.setdefault("ANTHROPIC_API_KEY", ${JSON.stringify(arkKey)})`,
    `os.environ.setdefault("ANTHROPIC_BASE_URL", ${JSON.stringify(arkBase)})`,
    `os.environ.setdefault("ANTHROPIC_MODEL", ${JSON.stringify(arkModel)})`,
    `os.environ.setdefault("QUANTAGENT_TRIAL_API_KEY", ${JSON.stringify(deepseekKey)})`,
    "",
  ].join("\n");

  writeFileSync(join(targetDir, "_builtin_keys.py"), py, "utf8");
  console.log(`Wrote _builtin_keys.py (ARK key length: ${arkKey.length})`);
}

function writeVariantFile(targetDir, variant) {
  const py = [
    '"""Build-time generated application variant. Do not edit; regenerated by prepare-backend.mjs."""',
    `APP_VARIANT = ${JSON.stringify(variant)}`,
    "",
  ].join("\n");
  writeFileSync(join(targetDir, "_variant.py"), py, "utf8");
  console.log(`Wrote _variant.py (variant: ${variant})`);
}

function findSitePackages() {
  const candidates = [join(projectRoot, ".venv/Lib/site-packages")];
  const unixVenvLib = join(projectRoot, ".venv/lib");

  if (existsSync(unixVenvLib)) {
    for (const entry of readdirSync(unixVenvLib, { withFileTypes: true })) {
      if (entry.isDirectory() && entry.name.startsWith("python")) {
        candidates.push(join(unixVenvLib, entry.name, "site-packages"));
      }
    }
  }

  const found = candidates.find((candidate) => existsSync(candidate));
  if (!found) {
    throw new Error(
      `Could not find .venv site-packages under ${projectRoot}. Checked:\n${candidates.join("\n")}`
    );
  }

  return found;
}

function findVenvPython() {
  const candidates = [
    join(projectRoot, ".venv/Scripts/python.exe"),
    join(projectRoot, ".venv/bin/python"),
  ];
  const found = candidates.find((candidate) => existsSync(candidate));

  if (!found) {
    throw new Error(`Could not find the .venv Python executable under ${projectRoot}`);
  }

  return found;
}

function findVenvBasePrefix() {
  const cfgPath = join(projectRoot, ".venv", "pyvenv.cfg");
  if (!existsSync(cfgPath)) {
    throw new Error(`Could not find ${cfgPath}`);
  }
  const text = readFileSync(cfgPath, "utf8");
  for (const line of text.split(/\r?\n/)) {
    const m = /^\s*home\s*=\s*(.+?)\s*$/.exec(line);
    if (m) {
      return m[1];
    }
  }
  throw new Error(`pyvenv.cfg at ${cfgPath} has no 'home' entry`);
}

function prepareWindowsPythonRuntime(systemPrefix) {
  const basePrefix = systemPrefix;

  mkdirSync(pythonRuntimeOut, { recursive: true });

  for (const dir of ["DLLs", "Lib"]) {
    const source = join(basePrefix, dir);
    if (!existsSync(source)) {
      continue;
    }
    cpDirRecursive(source, join(pythonRuntimeOut, dir), (entry) => {
      const normalized = entry.replaceAll("\\", "/").toLowerCase();
      return !normalized.includes("/lib/site-packages")
        && !normalized.includes("/__pycache__/")
        && !normalized.endsWith("/__pycache__")
        && !normalized.endsWith(".pyc");
    });
  }

  for (const entry of readdirSync(basePrefix, { withFileTypes: true })) {
    if (!entry.isFile()) {
      continue;
    }
    if (/^(python.*\.(exe|dll)|vcruntime.*\.dll|license.*)$/i.test(entry.name)) {
      cpSync(join(basePrefix, entry.name), join(pythonRuntimeOut, entry.name));
    }
  }

  if (!existsSync(join(pythonRuntimeOut, "python.exe"))) {
    throw new Error(`Could not prepare Windows Python runtime from ${basePrefix}`);
  }

  console.log(`Bundled Windows Python runtime: ${pythonRuntimeOut}`);
}
