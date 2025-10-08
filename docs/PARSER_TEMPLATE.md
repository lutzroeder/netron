# Parser template for Netron

This document explains how to add a new model parser to Netron. Parsers live in `source/` and follow a common ModelFactory interface used by `view.js` and the loader.

Key points
- Parsers are ES modules in `source/` named after the format (for example `onnx.js`, `tflite.js`).
- Prefer platform-agnostic code: use `Context.request()` / `Host.require()` for I/O rather than calling `fs` or `fetch` directly.
- Use `base.BinaryStream`, `base.Metadata` and existing helpers from `source/base.js`.

Minimal ModelFactory shape

A parser should export a ModelFactory class with at least these methods:

- async match(context)
  - Inspect `context.identifier`, `context.entries`, and `context.stream` to determine whether this parser can handle the input.
  - If matched, return a reader object via `context.set(name, reader)` or return a non-null value. If not, return null.

- async open(context)
  - Called after a successful match. Should read the `context.value` (often `target`) and return a Model-like object used by the view layer.
  - Many parsers call `await target.read()` if the target exposes a read method.

- filter(context, type) (optional)
  - Used to filter or transform types produced by the parser.

Example (minimal)

See `source/parser-template.js` for a minimal example. The template demonstrates:

- how to detect an extension (`.tpl`) in `match()`;
- how to call `await target.read()` in `open()` when a target provides `read()`;
- how to use `base.Metadata` for metadata scaffolding.

Concrete references in this repo
- `source/onnx.js` — large, real-world example of the `ModelFactory` pattern. Inspect `match`, `open`, and `filter` usage.
- `source/base.js` — contains `BinaryStream`, `Metadata`, and other helpers used throughout parsers.
- `source/index.js` — browser loader showing how modules are loaded and preloaded in the web build.
- `source/desktop.mjs` / `source/app.js` — Electron host and main process; check these when you need Node-specific features.

Common pitfalls
- Do not call Node `fs` or `http` directly in parser modules that are expected to run in the browser build. Use `Context.request()` or `Host.request()`.
- Keep module file names and metadata JSON (`*-metadata.json`) consistent. `base.Metadata().extensions` is used to assemble file filters.
- Avoid large synchronous file reads for >2GB files — use `node.FileStream` via the host to stream large files.

Testing parsers
- Add minimal smoke tests under `test/` that import your parser and validate the exported shape (see `test/parser-template.test.js`).
- For full parser verification, add fixtures to `test/fixtures/` and a conformance harness that exercises `match()` + `open()` + basic model introspection.

If you'd like, I can extend `parser-template.js` into a slightly more realistic example that parses a tiny header and add a conformance harness that runs all parsers against simple fixtures.
