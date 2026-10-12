// Generates the DUCO-S1 compute shader for one decimal nonce length.
//
// A nonce with D digits is split into a fixed high part (written by the CPU
// for each row of 10,000,000 nonces), up to six thread digits taken from the
// invocation index, and a last digit that each invocation loops over (0-9).
// Every byte position is known when the shader is generated, so the message
// stays in named registers: no arrays, no dynamic indexing and no spills.
//
// Each message and schedule word is classified as:
//   ROW    - the same for the whole row: the CPU folds it into constants,
//   THREAD - fixed per invocation: computed once, before the digit loop,
//   LOOP   - changes with the last digit j. The schedule is linear over XOR, so
//            W[t](j) = W[t](0) ^ E[t][j], where E is a constant: W[t](0) is computed
//            once per invocation and the loop only applies E.

// Bump when the generated shader changes, so saved GPU calibrations are measured again.
export const SHADER_VERSION = 3;
export const ROW_WORDS = 192; // 768 bytes, a multiple of the 256-byte offset alignment.
export const ROW_NONCES = 10000000;
export const IV = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];
const ROUND_K = [0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6];

// Row record layout (u32 words).
export const SLOT = {
  vFirst: 0, jFirst: 1, vLast: 2, jLast: 3,
  row: 4,      // index of the row in its batch, which selects its result slot
  target: 6,   // 6..10, expected digest
  a76: 11,     // expected value of A76, derived from the digest's last word
  state: 12,   // 12..16, a..e before round `start`
  word: 17,    // 17..32, message words with the fixed digits and padding
  xor: 17,     // xor[t] at 17 + t for t = 16..79: fixed part of schedule word t
  kw: 97,      // kw[t] at 97 + t: K + W[t] for fixed words
};

const ROW = 0;
const THREAD = 1;
const LOOP = 2;

export function roundK(t) { return ROUND_K[(t / 20) | 0]; }

export function rotl(value, bits) {
  return ((value << bits) | (value >>> (32 - bits))) >>> 0;
}

export function roundF(t, b, c, d) {
  if (t < 20) return (d ^ (b & (c ^ d))) >>> 0;
  if (t < 40 || t >= 60) return (b ^ c ^ d) >>> 0;
  return ((b & c) | (d & (b | c))) >>> 0;
}

// Byte positions and word classes for nonces with `digits` decimal digits.
export function shaderPlan(digits) {
  if (!Number.isInteger(digits) || digits < 1 || digits > 15) throw new Error("Unsupported nonce length.");
  const low = Math.min(digits, 7);
  const lastPosition = 40 + digits - 1;
  const threadPositions = [];
  for (let position = 40 + digits - low; position < lastPosition; position++) threadPositions.push(position);
  const cls = new Array(80).fill(ROW);
  for (const position of threadPositions) cls[position >> 2] = THREAD;
  cls[lastPosition >> 2] = LOOP;
  for (let t = 16; t < 80; t++) cls[t] = Math.max(cls[t - 3], cls[t - 8], cls[t - 14], cls[t - 16]);
  const start = cls.findIndex((value, index) => index >= 10 && value !== ROW);
  return { digits, low, lastPosition, threadPositions, cls, start, loopWord: lastPosition >> 2 };
}

const hex = value => `0x${(value >>> 0).toString(16)}u`;

// Shader variants for the last-digit loop. Both use the same arithmetic; GPUs differ in
// which one runs faster, so the miner times them on each GPU.
//   table:    one loop over the digit, E[t][j] read from a constant array,
//   unrolled: one copy of the loop body per digit, E[t][j] as immediates.
export const SHADER_VARIANTS = ["table", "unrolled"];

export function buildShader(digits, { workgroupSize = 128, variant = "table" } = {}) {
  if (!SHADER_VARIANTS.includes(variant)) throw new Error("Unknown shader variant.");
  const plan = shaderPlan(digits);
  const { cls, start, loopWord } = plan;
  // E[j][t]: what the last digit j adds to word t through the linear message schedule.
  const shift = 24 - 8 * (plan.lastPosition & 3);
  const E = Array.from({ length: 10 }, (_, j) => {
    const delta = new Array(80).fill(0);
    delta[loopWord] = j << shift;
    for (let t = 16; t < 80; t++) delta[t] = rotl((delta[t - 3] ^ delta[t - 8] ^ delta[t - 14] ^ delta[t - 16]) >>> 0, 1);
    return delta;
  });

  const pre = [];
  const loaded = new Set();
  const row = index => {
    if (!loaded.has(index)) {
      loaded.add(index);
      pre.push(`let r${index} = rows[rb + ${index}u];`);
    }
    return `r${index}`;
  };

  // Thread digits, most significant first.
  const digitNames = [];
  const count = plan.threadPositions.length;
  const smallDigits = (value, n, prefix) => {
    if (n === 3) {
      pre.push(`let ${prefix}0 = (${value} * 41u) >> 12u;`, `let ${prefix}r = ${value} - ${prefix}0 * 100u;`,
        `let ${prefix}1 = (${prefix}r * 103u) >> 10u;`, `let ${prefix}2 = ${prefix}r - ${prefix}1 * 10u;`);
      digitNames.push(`${prefix}0`, `${prefix}1`, `${prefix}2`);
    } else if (n === 2) {
      pre.push(`let ${prefix}0 = (${value} * 103u) >> 10u;`, `let ${prefix}1 = ${value} - ${prefix}0 * 10u;`);
      digitNames.push(`${prefix}0`, `${prefix}1`);
    } else if (n === 1) {
      digitNames.push(value);
    }
  };
  if (count > 3) {
    pre.push("let vh = v / 1000u;", "let vl = v - vh * 1000u;");
    smallDigits("vh", count - 3, "dh");
    smallDigits("vl", 3, "dl");
  } else {
    smallDigits("v", count, "dl");
  }

  // Message and schedule words that are not fixed for the row, with the last digit at 0.
  const x = [];
  for (let i = 10; i < 16; i++) {
    if (cls[i] === ROW) continue;
    let expression = row(SLOT.word + i);
    plan.threadPositions.forEach((position, n) => {
      if (position >> 2 === i) expression += ` | (${digitNames[n]} << ${24 - 8 * (position & 3)}u)`;
    });
    x[i] = `x${i}`;
    pre.push(`let ${x[i]} = ${expression};`);
  }
  for (let t = 16; t < 80; t++) {
    if (cls[t] === ROW) continue;
    const sources = [t - 3, t - 8, t - 14, t - 16];
    const terms = sources.filter(i => cls[i] !== ROW).map(i => x[i]);
    const fixed = sources.some(i => cls[i] === ROW) ? [row(SLOT.xor + t)] : [];
    x[t] = `x${t}`;
    pre.push(`let ${x[t]} = rol(${[...fixed, ...terms].join(" ^ ")}, 1u);`);
  }

  const fExpression = (t, p, q, r) => {
    if (t < 20) return `(${r} ^ (${p} & (${q} ^ ${r})))`;
    if (t < 40 || t >= 60) return `(${p} ^ ${q} ^ ${r})`;
    return `((${p} & ${q}) | (${r} & (${p} | ${q})))`;
  };
  const roundWord = (t, loopWord) => cls[t] === ROW ? row(SLOT.kw + t)
    : `${hex(roundK(t))} + ${cls[t] === THREAD ? x[t] : loopWord(t)}`;

  // State before round `start` comes from the CPU. Rounds before the first word that
  // depends on the last digit are computed once per thread, and so is everything in
  // that first round except its word.
  let a = row(SLOT.state), b = row(SLOT.state + 1), c = row(SLOT.state + 2), d = row(SLOT.state + 3), e = row(SLOT.state + 4);
  let first = start;
  for (; cls[first] !== LOOP; first++) {
    pre.push(`let a${first + 1} = rol(${a}, 5u) + ${fExpression(first, b, c, d)} + ${e} + ${roundWord(first)};`,
      `let c${first + 1} = rol(${b}, 30u);`);
    [a, b, c, d, e] = [`a${first + 1}`, a, `c${first + 1}`, c, d];
  }
  pre.push(`let p${first} = rol(${a}, 5u) + ${fExpression(first, b, c, d)} + ${e} + ${hex(roundK(first))};`,
    `let c${first + 1} = rol(${b}, 30u);`);
  const loopState = [`a${first + 1}`, a, `c${first + 1}`, c, d];

  // Rounds for one value of the last digit; `loopWord(t)` gives W[t] for that digit.
  // Rounds 76..79 only run when A76 already matches the target or for the sample.
  const iteration = (loopWord, indent, guard = null) => {
    const body = [`let a${first + 1} = p${first} + ${loopWord(first)};`];
    const tail = [];
    let [sa, sb, sc, sd, se] = loopState;
    let a76;
    for (let t = first + 1; t < 80; t++) {
      if (t === 76) a76 = sa;
      (t < 76 ? body : tail).push(`let a${t + 1} = rol(${sa}, 5u) + ${fExpression(t, sb, sc, sd)} + ${se} + ${roundWord(t, loopWord)};`,
        `let c${t + 1} = rol(${sb}, 30u);`);
      [sa, sb, sc, sd, se] = [`a${t + 1}`, sa, `c${t + 1}`, sc, sd];
    }
    const digest = [sa, sb, sc, sd, se];
    return `${body.join(`\n${indent}`)}
${indent}if (${guard ? `${guard} && (${a76} == expected || index == sampleIndex)` : `${a76} == expected || index == sampleIndex`}) {
${indent}  ${tail.join(`\n${indent}  `)}
${indent}  ${digest.map((value, i) => `let h${i} = ${hex(IV[i])} + ${value};`).join(`\n${indent}  `)}
${indent}  if (index == sampleIndex) {
${indent}    ${digest.map((_, i) => `result.sample[${i}] = h${i};`).join(`\n${indent}    `)}
${indent}  }
${indent}  if (${digest.map((_, i) => `h${i} == rows[rb + ${SLOT.target + i}u]`).join(" && ")}) {
${indent}    atomicMin(&result.found[row], index);
${indent}  }
${indent}}`;
  };

  let table = "";
  let loop;
  if (variant === "unrolled") {
    loop = E.map((delta, j) => `if (jFirst <= ${j}u && ${j}u <= jLast) {
    let index = offsetBase + ${j}u;
    ${iteration(t => delta[t] ? `(${x[t]} ^ ${hex(delta[t])})` : x[t], "    ")}
  }`).join("\n  ");
  } else {
    // Every invocation runs all ten digits so `j`, and the table address, stay uniform.
    const slots = [];
    for (let t = 0; t < 80; t++) if (cls[t] === LOOP) slots[t] = slots.filter(slot => slot !== undefined).length;
    const width = slots.filter(slot => slot !== undefined).length;
    const values = E.flatMap(delta => delta.filter((_, t) => slots[t] !== undefined).map(hex));
    table = `const E = array<u32, ${values.length}>(${values.join(", ")});\n`;
    loop = `for (var j = 0u; j < 10u; j = j + 1u) {
    let eb = j * ${width}u;
    let index = offsetBase + j;
    let inRange = jFirst <= j && j <= jLast;
    ${iteration(t => `(${x[t]} ^ E[eb + ${slots[t]}u])`, "    ", "inRange")}
  }`;
  }

  const code = `// Generated by sha1-shader.js (${variant}) for ${digits}-digit nonces.
@group(0) @binding(0) var<storage, read> rows: array<u32>;
// found[row] is the lowest matching offset within that row, or 0xffffffff.
struct Result { sample: array<u32, 5>, found: array<atomic<u32>> };
@group(0) @binding(1) var<storage, read_write> result: Result;
${table}
fn rol(value: u32, bits: u32) -> u32 { return (value << bits) | (value >> (32u - bits)); }

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let rb = wid.y * ${ROW_WORDS}u;
  let t = wid.x * ${workgroupSize}u + lid;
  let v = rows[rb + ${SLOT.vFirst}u] + t;
  let vLast = rows[rb + ${SLOT.vLast}u];
  if (v > vLast) { return; }
  let jFirst = select(0u, rows[rb + ${SLOT.jFirst}u], t == 0u);
  let jLast = select(9u, rows[rb + ${SLOT.jLast}u], v == vLast);
  let row = rows[rb + ${SLOT.row}u];
  let offsetBase = t * 10u - rows[rb + ${SLOT.jFirst}u];
  // Only the first nonce of the batch writes the sample digest.
  let sampleIndex = select(0xffffffffu, 0u, row == 0u);
  let expected = rows[rb + ${SLOT.a76}u];
  ${pre.join("\n  ")}
  ${loop}
}
`;
  return { code, plan };
}

// Fills one row record. `words` holds message words 0..15 with the fixed digits,
// the ASCII zero of every varying digit, the padding byte and the bit length.
export function packRow(out, at, plan, words, target, range) {
  const { cls, start } = plan;
  const w = new Array(80);
  for (let i = 0; i < 16; i++) w[i] = words[i] >>> 0;
  const record = out.subarray(at, at + ROW_WORDS);
  record.fill(0);
  for (let t = 16; t < 80; t++) {
    let fixed = 0;
    for (const i of [t - 3, t - 8, t - 14, t - 16]) if (cls[i] === ROW) fixed ^= w[i];
    if (cls[t] === ROW) w[t] = rotl(fixed >>> 0, 1);
    record[SLOT.xor + t] = fixed >>> 0;
  }
  for (let t = 0; t < 80; t++) if (cls[t] === ROW) record[SLOT.kw + t] = (roundK(t) + w[t]) >>> 0;
  let [a, b, c, d, e] = IV;
  for (let t = 0; t < start; t++) {
    const next = (rotl(a, 5) + roundF(t, b, c, d) + e + roundK(t) + w[t]) >>> 0;
    e = d;
    d = c;
    c = rotl(b, 30);
    b = a;
    a = next;
  }
  record.set([a, b, c, d, e], SLOT.state);
  for (let i = 0; i < 16; i++) record[SLOT.word + i] = w[i];
  record.set(target, SLOT.target);
  record[SLOT.a76] = rotl((target[4] - IV[4]) >>> 0, 2);
  record[SLOT.vFirst] = range.vFirst;
  record[SLOT.jFirst] = range.jFirst;
  record[SLOT.vLast] = range.vLast;
  record[SLOT.jLast] = range.jLast;
  record[SLOT.row] = range.row;
}
