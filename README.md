# DUCO GPU Lab

**[Open the live WebGPU miner](https://ajdasz.github.io/duco-gpu-lab/)**

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

## GitHub Pages

The live site is published from the `main` branch and `/(root)` through GitHub Pages. No build step is needed. The empty `.nojekyll` file tells Pages to serve the static files as they are, including the WebGPU shader.

GitHub Pages serves the project over HTTPS, which WebGPU needs outside localhost. Login and mining key values saved on localhost do not transfer to the Pages site because browser storage belongs to each site address. Each visitor enters and stores their own values in their browser. After publishing, verify that the pool accepts job requests and share submissions from the Pages address; the pool controls its cross-origin access policy.

## How it works

The pool sends a previous hash, target SHA-1 digest, and difficulty. The shader hashes `previous hash + decimal nonce` for each candidate from `0` through `100 × difficulty`. Each dispatch handles one batch of candidates. A found nonce is checked again with the browser's independent SHA-1 implementation before submission. The optional mining key is sent as `k` when a result is submitted. The Accepted and Rejected counters change only after the pool replies.

The shader performs one SHA-1 block. This covers Duino-Coin's 40-character previous hashes and decimal nonces up to 15 digits. A job outside this supported range stops with a clear error. If the full pool range has no match, the miner requests a new job without submitting an invalid nonce. GPU self-tests cover the old 32-bit boundary and larger nonces before connecting to the pool.

## Performance

The batch size starts at a conservative 16,776,960 candidates with **Auto** enabled. After three full batches at one size, Auto uses their median time to step up or down through the available sizes. It aims for 55 to 95 ms per batch, up to 536,862,720 candidates, and ignores shorter batches capped by the end of a pool job. Intermediate 100,661,760 and 201,323,520 candidate sizes give Auto finer steps around its target. Selecting a batch size manually turns Auto off; the checkbox turns it back on. **Batch time** appears under Connection details. Larger batches reduce submission and readback overhead but may compute more candidates after a match. Select a smaller size if Stop feels delayed or the browser becomes less responsive.

The SHA-1 shader is generated for each nonce length in [sha1-shader.js](sha1-shader.js) and keeps the whole message in registers. Work shared by many nonces is computed once, either on the CPU for each group of 10,000,000 nonces or once per GPU thread. Two batches stay queued, so the GPU does not wait while the browser reads a result.

The miner uses 256 threads per workgroup on NVIDIA adapters that support it and 128 elsewhere. On one RTX 3060 Laptop GPU, a local no-pool benchmark measured about 7.5 to 7.6 GH/s with 536,862,720 candidates per batch. Batches of 1 to 2 billion candidates did not improve sustained speed and took about 130 to 267 ms each, so they are not offered in the miner. These are local compute measurements, not pool-confirmed mining results.

The displayed hashrate measures candidate hashes computed during the current job. It includes all GPU invocations in a batch, even when a matching nonce is found before the end of that batch. The Hashes tried counter uses K, M, B, T, and Q suffixes; hover it to see the full count. Pool replies, job difficulty, and network time determine how often shares are accepted, so a synthetic GPU benchmark is not a promise of matching payout speed.

The header identifies the WebGPU adapter as precisely as the browser allows. Browsers may hide the exact model, so the page also checks the local WebGL renderer and shows its model when the vendor matches. On systems with multiple GPUs from the same vendor, WebGL may identify a different card; the tooltip says when WebGL supplied the model. If neither API provides a model, the header shows the available vendor and architecture. The GPU name stays in the browser and is not submitted to the pool.

## Tested GPUs

Approximate hashrates observed in the live miner. Browser, drivers, power limits, and batch size can change the result.

| GPU | Observed hashrate |
| --- | ---: |
| RTX 3060 (mobile) | ~3.3 GH/s* |
| Radeon 680M (mobile) | ~1.3 GH/s* |
| Intel UHD 620 (mobile) | ~150 MH/s |

\* Measured with an earlier, slower version of the shader.

Tested another GPU? [Open a pull request](https://github.com/ajdasz/duco-gpu-lab/pulls) with the GPU model, browser, batch size, and observed hashrate.

Protocol references: [official Mini Miner](https://server.duinocoin.com/miniminer.html) and [official PC miner](https://github.com/duino-coin/duino-coin/blob/master/PC_Miner.py).
