import { existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const python = existsSync(join(root, ".venv", "bin", "python"))
  ? join(root, ".venv", "bin", "python")
  : "python3";

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    env: process.env,
    ...options,
  });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function read(command, args) {
  return execFileSync(command, args, { cwd: root, encoding: "utf8" }).trim();
}

function targetTriple() {
  const explicit =
    process.env.TAURI_ENV_TARGET_TRIPLE ||
    process.env.CARGO_BUILD_TARGET ||
    process.env.TARGET;
  if (explicit) return explicit;

  const rustc = read("rustc", ["-vV"]);
  const host = rustc
    .split(/\r?\n/)
    .find((line) => line.startsWith("host:"))
    ?.replace("host:", "")
    .trim();
  if (!host) {
    throw new Error("Could not determine Rust target triple.");
  }
  return host;
}

function newestMtime(paths) {
  return Math.max(...paths.map((path) => statSync(path).mtimeMs));
}

const target = targetTriple();
const binariesDir = join(root, "src-tauri", "binaries");
const sidecar = join(binariesDir, `concept-api-${target}${target.includes("windows") ? ".exe" : ""}`);
const sources = [
  join(root, "concept_api.py"),
  join(root, "websearch.py"),
];

mkdirSync(binariesDir, { recursive: true });

if (
  existsSync(sidecar) &&
  statSync(sidecar).mtimeMs >= newestMtime(sources) &&
  process.argv.includes("--force") === false
) {
  console.log(`Python sidecar is up to date: ${sidecar}`);
  process.exit(0);
}

const pyinstallerCheck = spawnSync(python, ["-m", "PyInstaller", "--version"], {
  cwd: root,
  stdio: "ignore",
});
if (pyinstallerCheck.status !== 0) {
  run(python, ["-m", "pip", "install", "pyinstaller"]);
}

run(python, [
  "-m",
  "PyInstaller",
  "--noconfirm",
  "--clean",
  "--onefile",
  "--name",
  `concept-api-${target}`,
  "--distpath",
  binariesDir,
  "--workpath",
  join(root, ".sidecar-build", "work"),
  "--specpath",
  join(root, ".sidecar-build"),
  "--paths",
  root,
  "--hidden-import",
  "fitz",
  "--hidden-import",
  "bs4",
  "--hidden-import",
  "PIL",
  "--hidden-import",
  "PIL.Image",
  "--exclude-module",
  "torch",
  "--exclude-module",
  "transformers",
  "--exclude-module",
  "whisper",
  "--exclude-module",
  "cv2",
  join(root, "concept_api.py"),
]);
