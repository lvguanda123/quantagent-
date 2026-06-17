import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, "..");
const projectRoot = resolve(desktopDir, "../..");
const outputRoot = join(desktopDir, "bundled-backend");
const backendOut = join(outputRoot, "backend");
const pythonOut = join(outputRoot, "python");
const sitePackagesOut = join(pythonOut, "site-packages");

const backendFiles = [
  "agent_state.py",
  "color_style.py",
  "decision_agent.py",
  "default_config.py",
  "graph_setup.py",
  "graph_util.py",
  "indicator_agent.py",
  "pattern_agent.py",
  "proxy_fix.py",
  "requirements.txt",
  "static_util.py",
  "trading_graph.py",
  "trend_agent.py",
  "web_interface.py",
];

const backendDirs = ["assets", "data_providers", "static", "templates"];

const sitePackages = findSitePackages();

rmSync(outputRoot, { recursive: true, force: true });
mkdirSync(backendOut, { recursive: true });
mkdirSync(sitePackagesOut, { recursive: true });

for (const file of backendFiles) {
  copyRequired(join(projectRoot, file), join(backendOut, file));
}

for (const dir of backendDirs) {
  copyRequired(join(projectRoot, dir), join(backendOut, dir));
}

cpSync(sitePackages, sitePackagesOut, {
  recursive: true,
  filter: (source) => {
    const normalized = source.replaceAll("\\", "/");
    return !normalized.includes("/__pycache__/")
      && !normalized.endsWith("/__pycache__")
      && !normalized.endsWith(".pyc")
      && !normalized.includes("/pip/_vendor/cachecontrol/caches/");
  },
});

console.log(`Bundled backend: ${backendOut}`);
console.log(`Bundled Python site-packages: ${sitePackagesOut}`);

function copyRequired(source, target) {
  if (!existsSync(source)) {
    throw new Error(`Required backend path missing: ${source}`);
  }

  cpSync(source, target, { recursive: true });
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
