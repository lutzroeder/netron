# Embedding Netron

Netron's web build can be embedded in another page and handed a model
**directly as bytes**, over `window.postMessage`, without uploading it anywhere
or squeezing it into the URL. This mirrors
[Perfetto's embedding protocol](https://perfetto.dev/docs/visualization/embedding-the-ui)
and exists specifically to get around the browser URL-length limit (Chromium
blocks navigations to URLs longer than ~2&nbsp;MB, so the `#<data-url>` loading
path fails — `about:blank#blocked` — for any model over ~1.5&nbsp;MB).

## Protocol

1. Open Netron in a popup (`window.open`) or an `<iframe>`. Load it with the
   `?embed` query flag (e.g. `https://netron.app/?embed`) so it runs in
   **embedded mode** — the first-run update/cookie-consent dialogs and
   telemetry are suppressed, since an embedder is driving the window and a user
   may not be present to dismiss a dialog.
2. Post the string `'PING'` to it repeatedly. Netron replies with the string
   `'PONG'` once it is ready to receive a model. (It stays silent to pings sent
   before it is ready, so keep pinging on an interval until the first `'PONG'`.)
3. Once you receive `'PONG'`, stop pinging and post a single request:

   ```js
   target.postMessage({
     netron: {
       buffer,          // ArrayBuffer or typed array — the raw model bytes
       name: 'model.onnx', // display name; its extension drives format detection
       // identifier: 'model.onnx', // optional; defaults to `name`
       // url: 'https://…',          // optional; only used to resolve sidecar files
     }
   }, targetOrigin);
   ```

4. Netron opens the model and posts back
   `{ netron: { status: 'success', identifier } }` or
   `{ netron: { status: 'error', message } }`.

`buffer` is transferable — pass it in the second `postMessage` argument
(`postMessage(msg, targetOrigin, [buffer])`) to hand off large models without a
copy.

## Example — popup

```js
function openInNetron(bytes, name, netronOrigin = 'https://netron.app') {
  const win = window.open(netronOrigin);
  const onMessage = (e) => {
    if (e.source !== win) {
      return;
    }
    if (e.data === 'PONG') {
      clearInterval(ping);
      const buffer = bytes.buffer ? bytes.buffer : bytes; // accept a view or an ArrayBuffer
      win.postMessage({ netron: { buffer, name } }, netronOrigin, [buffer]);
    } else if (e.data && e.data.netron && e.data.netron.status) {
      window.removeEventListener('message', onMessage);
      if (e.data.netron.status === 'error') {
        console.error('Netron failed to open the model:', e.data.netron.message);
      }
    }
  };
  window.addEventListener('message', onMessage);
  const ping = setInterval(() => win.postMessage('PING', netronOrigin), 50);
}
```

## Example — iframe

```js
const frame = document.createElement('iframe');
frame.src = 'https://netron.app';
document.body.appendChild(frame);

const target = frame.contentWindow;
const onMessage = (e) => {
  if (e.source !== target) {
    return;
  }
  if (e.data === 'PONG') {
    clearInterval(ping);
    target.postMessage({ netron: { buffer, name: 'model.onnx' } }, '*', [buffer]);
  }
};
window.addEventListener('message', onMessage);
const ping = setInterval(() => target.postMessage('PING', '*'), 50);
```

## Notes

- **Nothing is uploaded.** The bytes travel in-process between windows of the
  same browser; Netron loads them from memory.
- **Origins.** Netron answers a ping to `event.source` using the ping's own
  `event.origin`, and accepts a model from any origin (it only *renders* a
  model, it never executes it). Send your model with the most specific
  `targetOrigin` you can rather than `'*'`.
- **Readiness.** Netron only answers `'PING'` after its view is initialized —
  which can be delayed by a first-run cookie-consent prompt — so the ping loop
  is required; a single ping may be missed.
