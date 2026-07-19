# QuantAgent macOS Desktop

This is the first-stage Tauri desktop shell for QuantAgent.

It does not start Flask automatically. Start the Flask server first, then run
the Tauri app.

## Requirements

- Node.js and npm
- Rust and Cargo
- Existing QuantAgent Python environment

## Development

From the project root:

```bash
python web_interface.py
```

In another terminal:

```bash
cd desktop/tauri
npm install
npm run dev
```

The desktop app opens `http://127.0.0.1:5000`.
