import { spawnSync } from "node:child_process";
import { cpSync, existsSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

const args = process.argv.slice(2);
const env = {
  ...process.env,
  CARGO_HOME: resolve(".cargo-home"),
};

if (args[0] === "dev" || args[0] === "build") {
  run("node", ["scripts/prepare-backend.mjs"]);
}

runNpx(["tauri", ...args], env);

if (args[0] === "build") {
  fixMacAppBundle();
  syncWindowsReleaseResources();
}

function fixMacAppBundle() {
  const appPath = resolve("src-tauri/target/release/bundle/macos/QuantAgent.app");
  if (!existsSync(appPath) || process.platform !== "darwin") {
    return;
  }

  syncMacBundledBackend(appPath);
  run("xattr", ["-cr", appPath]);
  run("codesign", ["--force", "--deep", "--sign", "-", appPath]);
  run("codesign", ["--verify", "--deep", "--strict", "--verbose=2", appPath]);
}

function syncMacBundledBackend(appPath) {
  const source = resolve("bundled-backend");
  const target = join(appPath, "Contents/Resources/_up_/bundled-backend");
  if (!existsSync(source)) {
    return;
  }

  rmSync(target, { recursive: true, force: true });
  cpSync(source, target, { recursive: true });
}

function syncWindowsReleaseResources() {
  if (process.platform !== "win32") {
    return;
  }

  const source = resolve("bundled-backend");
  if (!existsSync(source)) {
    return;
  }

  for (const releaseDir of [
    resolve("src-tauri/target/release"),
    resolve("src-tauri/target/x86_64-pc-windows-msvc/release"),
  ]) {
    if (!existsSync(releaseDir)) {
      continue;
    }
    for (const target of [
      join(releaseDir, "_up_/bundled-backend"),
      join(releaseDir, "bundled-backend"),
    ]) {
      rmSync(target, { recursive: true, force: true });
      cpSync(source, target, { recursive: true });
    }
  }
}

function run(command, commandArgs) {
  const result = spawnSync(command, commandArgs, {
    stdio: "inherit",
    shell: false,
  });

  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function runNpx(commandArgs, commandEnv) {
  const command = process.platform === "win32" ? "npx.cmd" : "npx";
  const result = spawnSync(command, commandArgs, {
    env: commandEnv,
    stdio: "inherit",
    shell: false,
  });

  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
