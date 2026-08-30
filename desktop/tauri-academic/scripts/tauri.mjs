import { readFileSync, writeFileSync, readdirSync, unlinkSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { cpSync, existsSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

const args = process.argv.slice(2);
const env = {
  ...process.env,
  CARGO_HOME: resolve(".cargo-home"),
};

// Infer the build variant from package.json "name" so we don't have to
// duplicate the value in every CI workflow. "quantagent-academic" /
// "quantagent-trader" both end with the variant identifier.
function detectVariant() {
  try {
    const pkg = JSON.parse(readFileSync(resolve("package.json"), "utf8"));
    const name = String(pkg.name || "");
    if (name.endsWith("-academic")) return "academic";
    if (name.endsWith("-trader")) return "trader";
  } catch (e) {
    // fall through
  }
  return process.env.QUANTAGENT_APP_VARIANT || "trader";
}

const buildVariant = detectVariant();

if (args[0] === "dev" || args[0] === "build") {
  run("node", ["scripts/prepare-backend.mjs", `--variant=${buildVariant}`]);
}

runTauri(args, env);

if (args[0] === "build") {
  fixMacAppBundle();
  syncWindowsReleaseResources();
  injectWebView2Loader();
}

// The webview2-com-sys crate builds WebView2Loader.dll into the cargo
// target/release directory, but the Tauri-generated NSIS installer.nsi
// only packages the main exe. Without the DLL beside it, the installed
// app fails to launch with "找不到 WebView2Loader.dll". Patch the
// generated installer.nsi to add a File line for the DLL, then re-run
// makensis so the bundle/nsis/*-setup.exe contains it.
function injectWebView2Loader() {
  if (process.platform !== "win32") {
    return;
  }

  const releaseDir = resolve("src-tauri/target/release");
  const dllSrc = join(releaseDir, "WebView2Loader.dll");
  if (!existsSync(dllSrc)) {
    console.log(`[inject] WebView2Loader.dll not found at ${dllSrc}, skipping`);
    return;
  }

  const nsiPath = join(releaseDir, "nsis/x64/installer.nsi");
  if (!existsSync(nsiPath)) {
    console.log(`[inject] installer.nsi not found at ${nsiPath}, skipping`);
    return;
  }

  const text = readFileSync(nsiPath, "utf8");
  if (text.includes("WebView2Loader.dll")) {
    console.log("[inject] installer.nsi already references WebView2Loader.dll");
    return;
  }

  const marker = 'File "${MAINBINARYSRCPATH}"';
  const idx = text.indexOf(marker);
  if (idx < 0) {
    console.log(`[inject] marker ${marker} not found in installer.nsi, skipping`);
    return;
  }
  const insertion = `${marker}\n  File "${dllSrc.replaceAll("\\", "\\\\")}"`;
  const patched = text.replace(marker, insertion);
  writeFileSync(nsiPath, patched, "utf8");
  console.log(`[inject] Patched ${nsiPath} to include WebView2Loader.dll`);

  // Find makensis (NSIS compiler). tauri-cli normally invokes it
  // directly, so it's shipped with the Tauri toolchain install under
  // %LocalAppData%\tauri\NSIS\Bin. Walk the standard locations.
  const candidates = [
    process.env.LOCALAPPDATA && join(process.env.LOCALAPPDATA, "tauri/NSIS/Bin/makensis.exe"),
    "C:/Users/Administrator/AppData/Local/tauri/NSIS/Bin/makensis.exe",
    "C:/Program Files (x86)/NSIS/makensis.exe",
  ].filter(Boolean);
  const makensis = candidates.find((p) => existsSync(p));
  if (!makensis) {
    console.error("[inject] makensis not found; install NSIS to bake the DLL");
    return;
  }

  const result = spawnSync(makensis, [nsiPath], { stdio: "inherit" });
  if (result.status !== 0) {
    console.error(`[inject] makensis exited with status ${result.status}`);
    return;
  }

  // makensis writes the configured OutFile (nsis-output.exe) next to
  // installer.nsi. Replace the bundle/nsis/*-setup.exe with it.
  const generated = join(releaseDir, "nsis/x64/nsis-output.exe");
  if (!existsSync(generated)) {
    console.error(`[inject] Expected ${generated} after makensis, not found`);
    return;
  }
  const bundleDir = join(releaseDir, "bundle/nsis");
  if (existsSync(bundleDir)) {
    for (const name of readdirSync(bundleDir)) {
      if (name.endsWith("-setup.exe")) {
        unlinkSync(join(bundleDir, name));
      }
    }
    cpSync(generated, join(bundleDir, `QuantAgent-${capitalize(buildVariant)}_0.1.0_x64-setup.exe`));
    console.log(`[inject] Replaced bundle installer with ${generated}`);
  }
}

function capitalize(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function fixMacAppBundle() {
  const appPath = resolve("src-tauri/target/release/bundle/macos/QuantAgent.app");
  if (!existsSync(appPath) || process.platform !== "darwin") {
    return;
  }

  syncMacBundledBackend(appPath);
  removeLegacyMacLaunchFlag(appPath);
  run("xattr", ["-cr", appPath]);
  run("codesign", ["--force", "--deep", "--sign", "-", appPath]);
  run("codesign", ["--verify", "--deep", "--strict", "--verbose=2", appPath]);
}

function removeLegacyMacLaunchFlag(appPath) {
  const plistPath = join(appPath, "Contents/Info.plist");
  for (const key of ["LSRequiresCarbon", "CSResourcesFileMapped"]) {
    runOptional("/usr/libexec/PlistBuddy", ["-c", `Delete :${key}`, plistPath]);
  }
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

function runOptional(command, commandArgs) {
  spawnSync(command, commandArgs, {
    stdio: "ignore",
    shell: false,
  });
}

function runTauri(commandArgs, commandEnv) {
  const result = spawnSync(process.execPath, ["node_modules/@tauri-apps/cli/tauri.js", ...commandArgs], {
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
