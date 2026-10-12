import { ROW_NONCES, ROW_WORDS, buildShader, packRow } from "./sha1-shader.js";

const BATCH_SIZE = 8388480 * 64;
// Rows of 10,000,000 nonces, plus the shorter rows below 10,000,000 split by digit count.
const MAX_ROWS = Math.ceil(BATCH_SIZE / ROW_NONCES) + 9;
const RESULT_WORDS = 6;
// 40 prefix bytes plus 15 decimal nonce digits still fit in one SHA-1 block.
const MAX_NONCE = 999999999999999;
const DEFAULT_WORKGROUP_SIZE = 128;

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
  constructor(device, { workgroupSize = DEFAULT_WORKGROUP_SIZE } = {}) {
    this.device = device;
    this.workgroupSize = workgroupSize;
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
      size: RESULT_WORDS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // Two readback buffers let the next batch run while the previous result is mapped.
    this.readbacks = [0, 1].map(() => ({
      buffer: device.createBuffer({
        size: RESULT_WORDS * 4,
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
    this.reset = new Uint32Array([0xffffffff, 0, 0, 0, 0, 0]);
  }

  static async create(selectedAdapter = null, options = {}) {
    if (!navigator.gpu) {
      throw new Error("WebGPU is unavailable. Open this page in a recent Chrome or Chromium on localhost.");
    }
    const adapter = selectedAdapter || await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter was found. Check browser GPU settings and drivers.");
    const device = await adapter.requestDevice();
    const vendor = (device.adapterInfo?.vendor || adapter.info?.vendor || "").toLowerCase();
    const supports256 = device.limits.maxComputeWorkgroupSizeX >= 256 &&
      device.limits.maxComputeInvocationsPerWorkgroup >= 256;
    const workgroupSize = options.workgroupSize ?? (vendor === "nvidia" && supports256 ? 256 : DEFAULT_WORKGROUP_SIZE);
    const hasher = new GpuHasher(device, { ...options, workgroupSize });
    hasher.adapterInfo = device.adapterInfo || adapter.info || null;
    // Compile every nonce length up front; the pool range always starts at nonce 0.
    await Promise.all(Array.from({ length: 15 }, (_, i) => hasher.pipeline(i + 1)));
    await hasher.selfTest();
    return hasher;
  }

  pipeline(digits) {
    if (!this.pipelines.has(digits)) {
      const { code, plan } = buildShader(digits, { workgroupSize: this.workgroupSize });
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
      promise.catch(() => this.pipelines.delete(digits));
      this.pipelines.set(digits, promise);
    }
    return this.pipelines.get(digits);
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
  hashBatch(prefix, target, start, count) {
    if (!Number.isSafeInteger(start) || start < 0 || start > MAX_NONCE ||
        !Number.isInteger(count) || count < 1 || count > BATCH_SIZE || start + count - 1 > MAX_NONCE) {
      return Promise.reject(new Error("Invalid GPU nonce range."));
    }
    const submitted = this.queueTail.then(() => this.submitBatch(prefix, target, start, count));
    this.queueTail = submitted.then(() => {}, () => {});
    return submitted.then(pending => pending.result);
  }

  async submitBatch(prefix, target, start, count) {
    const readback = this.readbacks[this.nextReadback];
    this.nextReadback = (this.nextReadback + 1) % this.readbacks.length;
    await readback.done;
    this.writeJob(prefix, target);
    const rows = splitRows(start, count);
    const dispatches = [];
    for (const [index, row] of rows.entries()) {
      const { pipeline, plan } = await this.pipeline(row.digits);
      packRow(this.rowData, index * ROW_WORDS, plan, this.rowWords(row.digits, row.high), this.targetWords, row);
      const invocations = row.vLast - row.vFirst + 1;
      const last = dispatches.at(-1);
      if (last && last.pipeline === pipeline) {
        last.rows++;
        last.invocations = Math.max(last.invocations, invocations);
      } else {
        dispatches.push({ pipeline, first: index, rows: 1, invocations });
      }
    }
    this.device.queue.writeBuffer(this.input, 0, this.rowData, 0, rows.length * ROW_WORDS);
    this.device.queue.writeBuffer(this.output, 0, this.reset);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const dispatch of dispatches) {
      pass.setPipeline(dispatch.pipeline);
      pass.setBindGroup(0, this.bindGroup, [dispatch.first * ROW_WORDS * 4]);
      pass.dispatchWorkgroups(Math.ceil(dispatch.invocations / this.workgroupSize), dispatch.rows);
    }
    pass.end();
    encoder.copyBufferToBuffer(this.output, 0, readback.buffer, 0, RESULT_WORDS * 4);
    this.device.queue.submit([encoder.finish()]);
    const result = readback.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const words = new Uint32Array(readback.buffer.getMappedRange()).slice();
      readback.buffer.unmap();
      return {
        found: words[0] === 0xffffffff ? null : start + words[0],
        sample: [...words.slice(1)].map(word => word.toString(16).padStart(8, "0")).join(""),
        count,
        start,
      };
    });
    readback.done = result.then(() => {}, () => {});
    return { result };
  }

  static get batchSize() { return BATCH_SIZE; }
  static get maxNonce() { return MAX_NONCE; }
}
