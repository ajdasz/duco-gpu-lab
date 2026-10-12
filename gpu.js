const ROW_SIZE = 8388480;
const BATCH_SIZE = ROW_SIZE * 64;
const RESULT_WORDS = 6;
const NONCE_BASE = 10000000;
// 40 prefix bytes plus 15 decimal nonce digits still fit in one SHA-1 block.
const MAX_NONCE = 999999999999999;

function rotateLeft(value, bits) {
  return ((value << bits) | (value >>> (32 - bits))) >>> 0;
}

function prefixState(words) {
  let a = 0x67452301;
  let b = 0xefcdab89;
  let c = 0x98badcfe;
  let d = 0x10325476;
  let e = 0xc3d2e1f0;
  for (let i = 0; i < 10; i++) {
    const next = (rotateLeft(a, 5) + ((b & c) | (~b & d)) + e + 0x5a827999 + words[i]) >>> 0;
    e = d;
    d = c;
    c = rotateLeft(b, 30);
    b = a;
    a = next;
  }
  return [a, b, c, d, e];
}

export class GpuHasher {
  constructor(device, pipeline) {
    this.device = device;
    this.pipeline = pipeline;
    this.input = device.createBuffer({
      size: 23 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.output = device.createBuffer({
      size: RESULT_WORDS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.readback = device.createBuffer({
      size: RESULT_WORDS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.input } },
        { binding: 1, resource: { buffer: this.output } },
      ],
    });
    this.cachedJob = "";
    this.range = new Uint32Array(3);
    this.reset = new Uint32Array([0xffffffff, 0, 0, 0, 0, 0]);
  }

  static async create() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is unavailable. Open this page in a recent Chrome or Chromium on localhost.");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter was found. Check browser GPU settings and drivers.");
    const device = await adapter.requestDevice();
    const response = await fetch("./gpu-sha1.wgsl", { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load the GPU shader. Serve this folder over localhost.");
    const module = device.createShaderModule({ code: await response.text() });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
    if (errors.length) throw new Error(`GPU shader error: ${errors[0].message}`);
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const hasher = new GpuHasher(device, pipeline);
    await hasher.selfTest();
    return hasher;
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
    ]) {
      const crossing = await this.hashBatch(previous, target, start, count);
      if (crossing.found !== expectedNonce) {
        throw new Error(`GPU nonce self-test failed across ${start}. Mining was not started.`);
      }
    }
    const secondRow = await this.hashBatch(previous, "68515ae22588424d90ff55fba7d34f2abac06c0e", 0, ROW_SIZE + 1);
    if (secondRow.found !== ROW_SIZE) {
      throw new Error("GPU multi-row nonce self-test failed. Mining was not started.");
    }
  }

  async hashBatch(prefix, target, start, count) {
    if (!Number.isSafeInteger(start) || start < 0 || start > MAX_NONCE ||
        !Number.isInteger(count) || count < 1 || count > BATCH_SIZE || start + count - 1 > MAX_NONCE) {
      throw new Error("Invalid GPU nonce range.");
    }
    const jobKey = `${prefix}:${target}`;
    if (jobKey !== this.cachedJob) {
      if (!/^[0-9a-f]{40}$/i.test(prefix)) throw new Error("Invalid previous hash from pool.");
      if (!/^[0-9a-f]{40}$/i.test(target)) throw new Error("Invalid target hash from pool.");
      const data = new Uint32Array(23);
      for (let i = 0; i < 10; i++) {
        const at = i * 4;
        data[i] = (prefix.charCodeAt(at) << 24) |
          (prefix.charCodeAt(at + 1) << 16) |
          (prefix.charCodeAt(at + 2) << 8) | prefix.charCodeAt(at + 3);
      }
      for (let i = 0; i < 5; i++) data[10 + i] = parseInt(target.slice(i * 8, i * 8 + 8), 16);
      data.set(prefixState(data), 18);
      this.device.queue.writeBuffer(this.input, 0, data);
      this.cachedJob = jobKey;
    }
    this.range[0] = Math.floor(start / NONCE_BASE);
    this.range[1] = start % NONCE_BASE;
    this.range[2] = count;
    this.device.queue.writeBuffer(this.input, 15 * 4, this.range);
    this.device.queue.writeBuffer(this.output, 0, this.reset);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    const workgroupsX = Math.min(65535, Math.ceil(count / 128));
    const workgroupsY = Math.ceil(count / ROW_SIZE);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY);
    pass.end();
    encoder.copyBufferToBuffer(this.output, 0, this.readback, 0, RESULT_WORDS * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(this.readback.getMappedRange()).slice();
    this.readback.unmap();
    return {
      found: words[0] === 0xffffffff ? null : start + words[0],
      sample: [...words.slice(1)].map(word => word.toString(16).padStart(8, "0")).join(""),
      count,
      start,
    };
  }

  static get batchSize() { return BATCH_SIZE; }
  static get maxNonce() { return MAX_NONCE; }
}
