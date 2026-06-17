import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const args = process.argv.slice(2);
const env = {
  ...process.env,
  CARGO_HOME: resolve(".cargo-home"),
};

const tauri = spawnSync("tauri", args, {
  env,
  stdio: "inherit",
  shell: false,
});

if (tauri.status !== 0) {
  process.exit(tauri.status ?? 1);
}

if (args[0] === "build") {
  fixMacAppBundle();
}

function fixMacAppBundle() {
  const appPath = resolve("src-tauri/target/release/bundle/macos/QuantAgent.app");
  if (!existsSync(appPath) || process.platform !== "darwin") {
    return;
  }

  run("xattr", ["-cr", appPath]);
  run("codesign", ["--force", "--deep", "--sign", "-", appPath]);
  run("codesign", ["--verify", "--deep", "--strict", "--verbose=2", appPath]);
}

function run(command, commandArgs) {
  const result = spawnSync(command, commandArgs, {
    stdio: "inherit",
    shell: false,
  });

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
