Netron — Copilot instructions

This file gives concise, actionable guidance for AI coding agents working on Netron (a model viewer supporting web, Electron and Python builds).

- Big picture
  - Netron is a client-side viewer with three packaging targets: web (`source/index.js`, `source/index.html`), Electron (`source/desktop.mjs`, `source/app.js`), and Python CLI (`onnx.py`, `package.py` + `server.py`).
  - Model format parsers and metadata live in `source/` as discrete modules named by format (for example, `onnx.js`, `tflite.js`, `pytorch.js`, `coreml.js`). Each parser typically exports a reader that accepts a `Context` or `BinaryStream`.
  - The UI/view layer is in `source/view.js` and `source/grapher.js`/`grapher.css`. `source/index.js` bootstraps the browser loader; `source/desktop.mjs` provides Electron Host/Context implementations.

- Important files to reference in edits
  - `package.json` / `package.js` — build, start, test commands and release workflows. Use `npm run start` (invokes `node package.js start`) or `node package.js build web|electron|python` for builds.
  - `source/index.js` — browser script loader and preload sequence. Follow its module loading pattern when adding new browser-only modules.
  - `source/desktop.mjs` and `source/app.js` — Electron host APIs and IPC channels (look for `ipcRenderer` / `ipcMain` handlers when changing host features).
  - Parsers: `source/*.js` (e.g., `onnx.js`, `tflite.js`, `pytorch.js`) — when modifying parsers, keep the same exported shapes and use `base.BinaryStream` and `base.Metadata` helpers from `source/base.js`.

- Conventions and patterns
  - Module loading: many source files are ES modules and are loaded dynamically by `window.exports.require(id)` in the browser build; keep side effects minimal and prefer exporting functions/classes.
  - Host abstraction: code should use `Context.request()` and `Host.require()` rather than direct fs/http when possible, to remain compatible with web and electron targets.
  - Telemetry and error reporting are centralized in `base.js` and `desktop.mjs`; avoid adding uncaught exceptions — surface errors via thrown Error objects with `context` when relevant.
  - Large file handling: desktop host uses `node.FileStream` to avoid reading >2GB into memory; when adding streaming parsers, support `BinaryStream` and `FileStream`.

- Build & test quick commands (verified from repo scripts)
  - Install deps: `npm install` (the project uses yarn historically but `package.js` calls `npm install` when node_modules is missing).
  - Start (Electron): `npm run start` (runs `node package.js start` → `npx electron .`).
  - Start the lightweight server (Python): `npm run server` (runs `python package.py build start`).
  - Build web bundle: `node package.js build web` or `npm run build` for full build (web + electron + python).
  - Run tests: `npm test` (runs `node test/models.js`).

- Small, actionable examples for common edits
  - Add a new parser module: create `source/<format>.js` exporting the same reader interface used by `onnx.js` (inspect `base.Metadata` and `view.js` for usage). Ensure browser compatibility (no Node-only APIs) or add Node-specific code behind Host abstractions in `desktop.mjs`/`node.js`.
  - Add a new configuration key: use `electron.ipcMain.on('get-configuration', ...)` / `set-configuration` in `source/app.js` and `Host.get(name)` / `Host.set(name, value)` in desktop/browser code.

- What to avoid / project-specific gotchas
  - Don't call fs/http directly in parser code; use the Host/Context request and require patterns to stay cross-platform.
  - Keep module filenames and exported names consistent with the format names in `base.Metadata().extensions` and the `source/*.json` metadata files.
  - When changing UI strings or titles, update `index.html` meta `version`/`date` replacements made by `package.js` during `build web`.

If anything in this guidance is unclear or you'd like additional examples (for example, a small parser template, or exact IPC message shapes), tell me which section to expand and I will iterate.
