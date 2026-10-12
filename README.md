# DUCO GPU Lab

An experimental, buildless Duino-Coin miner made out of curiosity. SHA-1 nonce candidates run in a WebGPU compute shader. The page displays actual GPU batch counts, hashrate, the current input and digest, and pool-confirmed accepted and rejected shares.

> Mining can increase power use and heat. Use DUCO GPU Lab at your own risk. This is an independent experimental project.

## Run

From this directory:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000` in a recent Chrome or Chromium with WebGPU enabled. Enter your Duino-Coin username, optionally enter a mining key if your account uses one, then press **Start mining**. The miner does not start automatically. Keep the tab active while mining. The username and mining key are saved in this browser's localStorage and restored after a refresh.

WebGPU requires a secure context. `localhost` is a secure context; opening `index.html` directly with a `file://` URL is not supported by this project.

The pool URL is editable under **More settings**. Its default is `https://server.duinocoin.com`, which exposes the same `/legacy_job` API used by the [official Mini Miner](https://server.duinocoin.com/miniminer.html). If Duino-Coin changes that API endpoint, update the field.

## Publish with GitHub Pages

This repository is ready for GitHub Pages without a build step. The empty `.nojekyll` file tells Pages to serve the static files as they are, including the WebGPU shader.

1. Create a public GitHub repository, for example `duco-gpu-lab`, and push these files to its `main` branch.
2. In the repository, open **Settings > Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**, then select `main` and `/(root)` and save.
4. Open `https://YOUR-USERNAME.github.io/duco-gpu-lab/` after Pages finishes publishing.

GitHub Pages serves the project over HTTPS, which WebGPU needs outside localhost. Login and mining key values saved on localhost do not transfer to the Pages site because browser storage belongs to each site address. Each visitor enters and stores their own values in their browser. After publishing, verify that the pool accepts job requests and share submissions from the Pages address; the pool controls its cross-origin access policy.

## How it works

The pool sends a previous hash, target SHA-1 digest, and difficulty. The shader hashes `previous hash + decimal nonce` for each candidate from `0` through `100 × difficulty`. Each dispatch handles one batch of candidates. A found nonce is checked again with the browser's independent SHA-1 implementation before submission. The optional mining key is sent as `k` when a result is submitted. The Accepted and Rejected counters change only after the pool replies.

The shader performs one SHA-1 block. This covers Duino-Coin's 40-character previous hashes and decimal nonces up to 15 digits. A job outside this supported range stops with a clear error. If the full pool range has no match, the miner requests a new job without submitting an invalid nonce. GPU self-tests cover the old 32-bit boundary and larger nonces before connecting to the pool.

## Performance

The batch size starts at a conservative 16,776,960 candidates with **Auto** enabled. After three full batches at one size, Auto uses their median time to step up or down through the available sizes. It aims for 35 to 65 ms per batch, up to 536,862,720 candidates, and ignores shorter batches capped by the end of a pool job. Selecting a batch size manually turns Auto off; the checkbox turns it back on. **Batch time** appears under Connection details. Larger batches reduce submission and readback overhead but may compute more candidates after a match. Select a smaller size if Stop feels delayed or the browser becomes less responsive.

The shader packs the fixed 40-character prefix once per job. The CPU precomputes the first ten SHA-1 rounds, which are identical for every nonce in a job. The GPU runs the remaining rounds in separate SHA-1 phases and keeps only 16 schedule words per invocation. The performance chart refreshes once per second and the detailed hash display refreshes at most five times per second. Calculation details are available in the expandable job section.

The displayed hashrate measures candidate hashes computed during the current job. It includes all GPU invocations in a batch, even when a matching nonce is found before the end of that batch. The Hashes tried counter uses K, M, B, T, and Q suffixes; hover it to see the full count. Pool replies, job difficulty, and network time determine how often shares are accepted, so a synthetic GPU benchmark is not a promise of matching payout speed.

Protocol references: [official Mini Miner](https://server.duinocoin.com/miniminer.html) and [official PC miner](https://github.com/duino-coin/duino-coin/blob/master/PC_Miner.py).
