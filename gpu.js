import { ROW_NONCES, ROW_WORDS, SHADER_VARIANTS, SHADER_VERSION, buildShader, packRow } from "./sha1-shader.js";

// Every row reports its own match, so the batch size is limited only by the row buffer.
// The largest batch leaves room for GPUs many times faster than an RTX 3060 Laptop GPU.
const BATCH_SIZE = 8388480 * 2048;
// Rows of 10,000,000 nonces, plus the shorter rows below 10,000,000 split by digit count.
const MAX_ROWS = Math.ceil(BATCH_SIZE / ROW_NONCES) + 9;
// Windows resets a GPU that runs one submission for about 2 seconds, so a batch is
// submitted as separate dispatches of at most 32 rows (320,000,000 nonces).
const MAX_DISPATCH_ROWS = 32;
const SAMPLE_WORDS = 5;
const NOT_FOUND = 0xffffffff;
// Readback layout: two pass timestamps, the sample digest, then one match offset per row.
const TIMESTAMP_BYTES = 16;
// 40 prefix bytes plus 15 decimal nonce digits still fit in one SHA-1 block.
const MAX_NONCE = 999999999999999;
// A kernel is a shader variant with a workgroup size.
const DEFAULT_KERNEL = { variant: "table", workgroupSize: 128 };
const WORKGROUP_SIZES = [64, 128, 256];
// Calibration warms the GPU up, then times rounds of one batch of about 40 ms per
// kernel. Another kernel must be at least 3% faster to replace the default.
const CALIBRATION = { warmUpMs: 400, batchMs: 40, rounds: 6, deadlineMs: 5000, minGain: 1.03 };
const CALIBRATION_JOB = ["7a8220fede52c1ccc2ee703cd1b51d36ec7e1ef8", "0".repeat(40)];
// Calibration batches start at the first 10-digit nonce and stay within 10 digits.
const CALIBRATION_START = 1000000000;
const CALIBRATION_MAX_COUNT = 9000000000;

function settingsKey(info) {
  const id = ["vendor", "architecture", "description"].map(field => info?.[field]?.trim() || "").join("|");
  return `duco-gpu-lab.gpu.v${SHADER_VERSION}.${id}`;
}

// Calibration and batch size saved for this GPU, browser and shader version,
// or null when browser storage is unavailable.
export function loadGpuSettings(info) {
  let text;
  try { text = localStorage.getItem(settingsKey(info)); } catch { return null; }
  try {
    const saved = JSON.parse(text);
    return saved && typeof saved === "object" ? saved : {};
  } catch { return {}; }
}

export function saveGpuSettings(info, values) {
  try {
    localStorage.setItem(settingsKey(info), JSON.stringify({ ...loadGpuSettings(info), ...values }));
  } catch { /* Without storage the GPU is calibrated again next time. */ }
}

const kernelKey = kernel => `${kernel.variant}:${kernel.workgroupSize}`;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

// Splits [start, start + count) into rows that share the nonce length and every digit
// except the last seven. The shader loops over the last digit; the other six come
// from the invocation index `v`.
export function splitRows(start, count) {
  const rows = [];
  const end = start + count - 1;
  let nonce = start;
  while (nonce <= end) {
    let digits, high, base, rowEnd;
    if (nonce < ROW_NONCES) {
      digits = String(nonce).length;
      high = "";
      base = 0;
      rowEnd = 10 ** digits - 1;
    } else {
      const highValue = Math.floor(nonce / ROW_NONCES);
      high = String(highValue);
      digits = high.length + 7;
      base = highValue * ROW_NONCES;
      rowEnd = base + ROW_NONCES - 1;
    }
    rowEnd = Math.min(rowEnd, end);
    rows.push({
      digits, high, offset: nonce - start,
      vFirst: Math.floor((nonce - base) / 10), jFirst: (nonce - base) % 10,
      vLast: Math.floor((rowEnd - base) / 10), jLast: (rowEnd - base) % 10,
    });
    nonce = rowEnd + 1;
  }
  return rows;
}

export class GpuHasher {
  constructor(device, { kernel = DEFAULT_KERNEL } = {}) {
    this.device = device;
    this.kernel = kernel;
    this.pipelines = new Map();
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", hasDynamicOffset: true } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.rowData = new Uint32Array(MAX_ROWS * ROW_WORDS);
    // Twice the rows, so a dispatch bound at any row offset still has a full-size binding.
    this.input = device.createBuffer({
      size: this.rowData.byteLength * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.output = device.createBuffer({
      size: (SAMPLE_WORDS + MAX_ROWS) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // GPU timestamps make calibration steadier when the browser exposes them.
    if (device.features.has("timestamp-query")) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: 4 });
      this.queryResolve = device.createBuffer({
        size: 512,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
    // Two readback buffers let the next batch run while the previous result is mapped.
    this.readbacks = [0, 1].map(() => ({
      buffer: device.createBuffer({
        size: TIMESTAMP_BYTES + (SAMPLE_WORDS + MAX_ROWS) * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      done: Promise.resolve(),
    }));
    this.nextReadback = 0;
    this.queueTail = Promise.resolve();
    this.bindGroup = device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.input, size: this.rowData.byteLength } },
        { binding: 1, resource: { buffer: this.output } },
      ],
    });
    this.cachedJob = "";
    this.reset = new Uint32Array(SAMPLE_WORDS + MAX_ROWS).fill(NOT_FOUND, SAMPLE_WORDS);
  }

  // Pass `kernel` to skip calibration; `onCalibrate` is called before a calibration starts.
  static async create(selectedAdapter = null, { kernel, onCalibrate } = {}) {
    if (!navigator.gpu) {
      throw new Error("WebGPU is unavailable. Open this page in a recent Chrome or Chromium on localhost.");
    }
    const adapter = selectedAdapter || await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter was found. Check browser GPU settings and drivers.");
    const info = adapter.info || null;
    // Without storage the calibration could not be kept, so the default kernel is used.
    const saved = loadGpuSettings(info);
    const needsCalibration = !kernel && saved !== null && !saved.variant;
    // Timestamps are only read during calibration, but requesting them every time keeps
    // the device the same, so the browser can reuse the compiled shaders on later visits.
    const timing = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice(timing ? { requiredFeatures: ["timestamp-query"] } : undefined);
    const kernels = SHADER_VARIANTS.flatMap(variant => WORKGROUP_SIZES
      .filter(size => size <= device.limits.maxComputeInvocationsPerWorkgroup && size <= device.limits.maxComputeWorkgroupSizeX)
      .map(workgroupSize => ({ variant, workgroupSize })));
    const savedKernel = kernel ? undefined : kernels.find(candidate => candidate.variant === saved?.variant &&
      candidate.workgroupSize === saved?.workgroupSize);
    const hasher = new GpuHasher(device, { kernel: kernel ?? savedKernel ?? DEFAULT_KERNEL });
    hasher.adapterInfo = info;
    hasher.calibration = {
      source: kernel ? "manual" : savedKernel ? "saved" : "default",
      rate: savedKernel && Number.isFinite(saved.rate) ? saved.rate : null,
    };
    if (needsCalibration && kernels.length > 1) {
      onCalibrate?.();
      try {
        // Only the winner gets every nonce length compiled and the full self-tests.
        const { kernel: best, rate } = await hasher.calibrate(kernels);
        hasher.kernel = best;
        await hasher.prepare();
        hasher.calibration = { source: "calibrated", rate };
        saveGpuSettings(info, { ...best, rate });
      } catch {
        hasher.kernel = DEFAULT_KERNEL;
      }
      for (const key of hasher.pipelines.keys()) {
        if (!key.startsWith(`${kernelKey(hasher.kernel)}:`)) hasher.pipelines.delete(key);
      }
    }
    if (hasher.calibration.source !== "calibrated") await hasher.prepare();
    return hasher;
  }

  get workgroupSize() { return this.kernel.workgroupSize; }

  // Compiles every nonce length for the current kernel and runs the self-tests.
  // The pool range always starts at nonce 0.
  async prepare() {
    await Promise.all(Array.from({ length: 15 }, (_, i) => this.pipeline(i + 1)));
    await this.selfTest();
  }

  // Times batches of 10-digit nonces with two queued, as in mining, so the GPU never
  // idles. Each batch's time comes from GPU timestamps or, without them, from the gap
  // between its result and the previous one. After a warm-up that sizes the batches,
  // every round runs one batch per kernel.
  // Returns the kernel with the best median rate, in hashes per second.
  async calibrate(kernels) {
    await Promise.all(kernels.map(kernel => this.pipeline(10, kernel)));
    const started = performance.now();
    const warmUpEnd = started + CALIBRATION.warmUpMs;
    const order = Array.from({ length: CALIBRATION.rounds }, () => kernels).flat();
    const samples = new Map(kernels.map(kernel => [kernel, []]));
    const queue = [];
    let count = 1 << 20;
    let measuring = false;
    const submit = () => {
      const kernel = measuring ? order.shift() : this.kernel;
      const result = this.hashBatch(...CALIBRATION_JOB, CALIBRATION_START, count, { kernel, timed: true });
      queue.push({ kernel, count, measured: measuring, result });
    };
    submit();
    submit();
    let last = null;
    while (queue.length) {
      const batch = queue.shift();
      const { gpuMs } = await batch.result;
      const now = performance.now();
      if (now - started > CALIBRATION.deadlineMs) throw new Error("GPU calibration took too long.");
      if (last !== null) {
        const ms = gpuMs ?? now - last;
        if (batch.measured) {
          samples.get(batch.kernel).push(batch.count / ms);
        } else if (!measuring) {
          count = Math.min(Math.max(Math.round(batch.count / ms * CALIBRATION.batchMs), 1 << 16), CALIBRATION_MAX_COUNT);
          measuring = now >= warmUpEnd && Math.abs(ms - CALIBRATION.batchMs) < CALIBRATION.batchMs / 4;
        }
      }
      last = now;
      if (!measuring || order.length) submit();
    }
    const rates = new Map([...samples].map(([kernel, values]) => [kernel, median(values)]));
    const fallback = kernels.find(kernel => kernelKey(kernel) === kernelKey(this.kernel));
    let best = fallback;
    for (const [kernel, rate] of rates) if (rate > rates.get(best)) best = kernel;
    if (rates.get(best) < rates.get(fallback) * CALIBRATION.minGain) best = fallback;
    return { kernel: best, rate: Math.round(rates.get(best) * 1000), rates };
  }

  pipeline(digits, kernel = this.kernel) {
    const key = `${kernelKey(kernel)}:${digits}`;
    if (!this.pipelines.has(key)) {
      const { code, plan } = buildShader(digits, kernel);
      const promise = (async () => {
        const module = this.device.createShaderModule({ code });
        const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
        if (errors.length) throw new Error(`GPU shader error: ${errors[0].message}`);
        const pipeline = await this.device.createComputePipelineAsync({
          layout: this.pipelineLayout,
          compute: { module, entryPoint: "main" },
        });
        return { pipeline, plan };
      })();
      promise.catch(() => this.pipelines.delete(key));
      this.pipelines.set(key, promise);
    }
    return this.pipelines.get(key);
  }

  async selfTest() {
    const expected = "eef42ae8318b13a6dc80d10bd8d7fa39e7525844"; // SHA-1(40 zeroes + "0")
    const test = await this.hashBatch("0".repeat(40), expected, 0, 1);
    if (test.found !== 0 || test.sample !== expected) {
      throw new Error("GPU SHA-1 self-test failed. Mining was not started.");
    }
    // Captured real pool job: a 40-byte prefix and a two-digit matching nonce.
    const previous = "7a8220fede52c1ccc2ee703cd1b51d36ec7e1ef8";
    const target = "51bb236f06a54db359ebf29f0983cd0fbc6e843e";
    const poolTest = await this.hashBatch(previous, target, 0, 80);
    if (poolTest.found !== 79 || poolTest.sample !== "f2bbcd87d985c1658c447e4bc873242265552004") {
      throw new Error("GPU pool-job self-test failed. Mining was not started.");
    }
    for (const [nonce, digest] of [
      [10, "59de27ef73b82e7b995cc11c2f36b7573a88c8b6"],
      [100, "dab05b4e3e63a1a15e5636ff5245bde14b296ea5"],
      [1000000000, "bfc88768199b1a54aaa40785d5e1838ce65d885e"],
      [4294967294, "0957147e40051187bc5fa9cb3397c2947c4c605b"],
      [4294967295, "8f7ead0fa28fc0adf30c1ac399267bdba09224a1"],
      [10000000, "bc8d8163441551786afeb34058d7beff1b4688e7"],
      [88612054500, "4d0b9d246d829d0afc72611656abffa8d4c1fe8b"],
      [MAX_NONCE, "973c30d53f2b0f6021bd6f753b417fafe2927035"],
    ]) {
      const boundary = await this.hashBatch(previous, digest, nonce, 1);
      if (boundary.found !== nonce || boundary.sample !== digest) {
        throw new Error(`GPU nonce self-test failed at ${nonce}. Mining was not started.`);
      }
    }
    for (const [start, count, target, expectedNonce] of [
      [9999997, 5, "bc8d8163441551786afeb34058d7beff1b4688e7", 10000000],
      [4294967293, 5, "8f7ead0fa28fc0adf30c1ac399267bdba09224a1", 4294967295],
      // Several rows and nonce lengths in one batch; the match is in a later row.
      [9999990, 30000020, "4d0b9d246d829d0afc72611656abffa8d4c1fe8b", null],
      [88600000000, 20000001, "4d0b9d246d829d0afc72611656abffa8d4c1fe8b", 88612054500],
      [99, 2, "dab05b4e3e63a1a15e5636ff5245bde14b296ea5", 100],
    ]) {
      const crossing = await this.hashBatch(previous, target, start, count);
      if (crossing.found !== expectedNonce) {
        throw new Error(`GPU nonce self-test failed across ${start}. Mining was not started.`);
      }
    }
  }

  writeJob(prefix, target) {
    const jobKey = `${prefix}:${target}`;
    if (jobKey === this.cachedJob) return;
    if (!/^[0-9a-f]{40}$/i.test(prefix)) throw new Error("Invalid previous hash from pool.");
    if (!/^[0-9a-f]{40}$/i.test(target)) throw new Error("Invalid target hash from pool.");
    this.prefixWords = new Uint32Array(16);
    for (let i = 0; i < 10; i++) {
      const at = i * 4;
      this.prefixWords[i] = (prefix.charCodeAt(at) << 24) |
        (prefix.charCodeAt(at + 1) << 16) |
        (prefix.charCodeAt(at + 2) << 8) | prefix.charCodeAt(at + 3);
    }
    this.targetWords = Array.from({ length: 5 }, (_, i) => parseInt(target.slice(i * 8, i * 8 + 8), 16));
    this.cachedJob = jobKey;
  }

  // Message words for one row: fixed high digits, ASCII "0" for the digits the shader
  // fills in, the padding byte and the message length in bits.
  rowWords(digits, high) {
    const words = this.prefixWords.slice();
    const text = high.padEnd(digits, "0");
    for (let i = 0; i <= digits; i++) {
      const position = 40 + i;
      const byte = i < digits ? text.charCodeAt(i) : 0x80;
      words[position >> 2] |= byte << (24 - 8 * (position & 3));
    }
    words[15] = (40 + digits) * 8;
    return words;
  }

  // Submits one batch and resolves with its result. Calls may overlap: batches are
  // submitted in call order, so the caller can queue the next batch before awaiting.
  // `timed` adds the GPU time of the batch as `gpuMs` when timestamps are available.
  hashBatch(prefix, target, start, count, { kernel = this.kernel, timed = false } = {}) {
    if (!Number.isSafeInteger(start) || start < 0 || start > MAX_NONCE ||
        !Number.isInteger(count) || count < 1 || count > BATCH_SIZE || start + count - 1 > MAX_NONCE) {
      return Promise.reject(new Error("Invalid GPU nonce range."));
    }
    const submitted = this.queueTail.then(() => this.submitBatch(prefix, target, start, count, kernel, timed));
    this.queueTail = submitted.then(() => {}, () => {});
    return submitted.then(pending => pending.result);
  }

  async submitBatch(prefix, target, start, count, kernel, timed) {
    const slot = this.nextReadback;
    const readback = this.readbacks[slot];
    this.nextReadback = (this.nextReadback + 1) % this.readbacks.length;
    await readback.done;
    this.writeJob(prefix, target);
    const rows = splitRows(start, count);
    const dispatches = [];
    for (const [index, row] of rows.entries()) {
      const { pipeline, plan } = await this.pipeline(row.digits, kernel);
      packRow(this.rowData, index * ROW_WORDS, plan, this.rowWords(row.digits, row.high), this.targetWords, { ...row, row: index });
      const invocations = row.vLast - row.vFirst + 1;
      const last = dispatches.at(-1);
      if (last && last.pipeline === pipeline && last.rows < MAX_DISPATCH_ROWS) {
        last.rows++;
        last.invocations = Math.max(last.invocations, invocations);
      } else {
        dispatches.push({ pipeline, first: index, rows: 1, invocations });
      }
    }
    this.device.queue.writeBuffer(this.input, 0, this.rowData, 0, rows.length * ROW_WORDS);
    const resultWords = SAMPLE_WORDS + rows.length;
    this.device.queue.writeBuffer(this.output, 0, this.reset, 0, resultWords);
    const stamps = timed && this.querySet ? slot * 2 : null;
    for (const [index, dispatch] of dispatches.entries()) {
      const first = index === 0;
      const last = index === dispatches.length - 1;
      const encoder = this.device.createCommandEncoder();
      // The timestamps span the whole batch: the first pass starts it, the last ends it.
      const pass = encoder.beginComputePass(stamps === null ? undefined : {
        timestampWrites: {
          querySet: this.querySet,
          ...(first && { beginningOfPassWriteIndex: stamps }),
          ...(last && { endOfPassWriteIndex: stamps + 1 }),
        },
      });
      pass.setPipeline(dispatch.pipeline);
      pass.setBindGroup(0, this.bindGroup, [dispatch.first * ROW_WORDS * 4]);
      pass.dispatchWorkgroups(Math.ceil(dispatch.invocations / kernel.workgroupSize), dispatch.rows);
      pass.end();
      if (last) {
        encoder.copyBufferToBuffer(this.output, 0, readback.buffer, TIMESTAMP_BYTES, resultWords * 4);
        if (stamps !== null) {
          encoder.resolveQuerySet(this.querySet, stamps, 2, this.queryResolve, slot * 256);
          encoder.copyBufferToBuffer(this.queryResolve, slot * 256, readback.buffer, 0, 16);
        }
      }
      this.device.queue.submit([encoder.finish()]);
    }
    const mappedBytes = TIMESTAMP_BYTES + resultWords * 4;
    const result = readback.buffer.mapAsync(GPUMapMode.READ, 0, mappedBytes).then(() => {
      const mapped = readback.buffer.getMappedRange(0, mappedBytes);
      const [begin, end] = new BigUint64Array(mapped, 0, 2);
      const words = new Uint32Array(mapped, TIMESTAMP_BYTES, resultWords).slice();
      readback.buffer.unmap();
      // Rows are in nonce order, so the first row with a match holds the lowest nonce.
      const hit = rows.findIndex((_, index) => words[SAMPLE_WORDS + index] !== NOT_FOUND);
      return {
        found: hit < 0 ? null : start + rows[hit].offset + words[SAMPLE_WORDS + hit],
        sample: [...words.slice(0, SAMPLE_WORDS)].map(word => word.toString(16).padStart(8, "0")).join(""),
        count,
        start,
        gpuMs: stamps !== null && end > begin ? Number(end - begin) / 1e6 : null,
      };
    });
    readback.done = result.then(() => {}, () => {});
    return { result };
  }

  static get batchSize() { return BATCH_SIZE; }
  static get maxNonce() { return MAX_NONCE; }
}
