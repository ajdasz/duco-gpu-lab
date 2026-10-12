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
//   LOOP   - changes with the last digit: computed inside the loop.

export const ROW_WORDS = 192; // 768 bytes, a multiple of the 256-byte offset alignment.
export const ROW_NONCES = 10000000;
export const IV = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];
const ROUND_K = [0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6];

// Row record layout (u32 words).
export const SLOT = {
  vFirst: 0, jFirst: 1, vLast: 2, jLast: 3, offset: 4,
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

export function buildShader(digits, { workgroupSize = 128 } = {}) {
  const plan = shaderPlan(digits);
  const { cls, start } = plan;
  const pre = [];
  const body = [];
  const tail = [];
  const loaded = new Set();
  const row = index => {
    if (!loaded.has(index)) {
      loaded.add(index);
      pre.push(`let r${index} = rows[rb + ${index}u];`);
    }
    return `r${index}`;
  };

  // Each value tracks where it was computed so loop-invariant work stays before the loop.
  const where = new Map();

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

  // Message words 10..15 that are not fixed for the row.
  const words = [];
  for (let i = 10; i < 16; i++) {
    if (cls[i] === ROW) continue;
    let expression = row(SLOT.word + i);
    plan.threadPositions.forEach((position, n) => {
      if (position >> 2 === i) expression += ` | (${digitNames[n]} << ${24 - 8 * (position & 3)}u)`;
    });
    if (cls[i] === THREAD) {
      pre.push(`let w${i} = ${expression};`);
    } else {
      pre.push(`let wb${i} = ${expression};`);
      body.push(`let w${i} = wb${i} | (j << ${24 - 8 * (plan.lastPosition & 3)}u);`);
    }
    words[i] = `w${i}`;
    where.set(words[i], cls[i]);
  }
  const out = (target, name, expression, level) => {
    target.push(`let ${name} = ${expression};`);
    where.set(name, level);
    return name;
  };
  const levelOf = name => where.get(name) ?? ROW;
  const sink = level => level === LOOP ? body : pre;

  const schedule = (t, target) => {
    const terms = [t - 3, t - 8, t - 14, t - 16].filter(i => cls[i] !== ROW).map(i => words[i]);
    const fixed = [t - 3, t - 8, t - 14, t - 16].some(i => cls[i] === ROW) ? row(SLOT.xor + t) : null;
    const invariant = terms.filter(name => levelOf(name) !== LOOP);
    const varying = terms.filter(name => levelOf(name) === LOOP);
    let inner = [fixed, ...invariant].filter(Boolean);
    if (cls[t] === LOOP && varying.length && inner.length > 1) {
      inner = [out(pre, `q${t}`, inner.join(" ^ "), THREAD)];
    }
    const all = [...inner, ...varying].join(" ^ ");
    const destination = target ?? sink(cls[t]);
    words[t] = out(destination, `w${t}`, `rol(${all}, 1u)`, target === tail ? LOOP : cls[t]);
  };

  // State before round `start` comes from the CPU.
  let a = row(SLOT.state);
  let b = row(SLOT.state + 1);
  let c = row(SLOT.state + 2);
  let d = row(SLOT.state + 3);
  let e = row(SLOT.state + 4);
  const fExpression = (t, x, y, z) => {
    if (t < 20) return `(${z} ^ (${x} & (${y} ^ ${z})))`;
    if (t < 40 || t >= 60) return `(${x} ^ ${y} ^ ${z})`;
    return `((${x} & ${y}) | (${z} & (${x} | ${y})))`;
  };
  const round = (t, target) => {
    if (t >= 16 && cls[t] !== ROW && words[t] === undefined) schedule(t, target);
    const stateLevel = Math.max(levelOf(a), levelOf(b), levelOf(c), levelOf(d), levelOf(e));
    const wordLevel = cls[t] === ROW ? ROW : levelOf(words[t]);
    const level = target === tail ? LOOP : Math.max(stateLevel, wordLevel);
    const destination = target ?? sink(level);
    const base = `rol(${a}, 5u) + ${fExpression(t, b, c, d)} + ${e}`;
    let next;
    if (cls[t] === ROW) {
      next = out(destination, `a${t + 1}`, `${base} + ${row(SLOT.kw + t)}`, level);
    } else if (stateLevel !== LOOP && wordLevel === LOOP && target !== tail) {
      // Everything except the varying word is loop-invariant.
      const partial = out(pre, `p${t}`, `${base} + ${hex(roundK(t))}`, THREAD);
      next = out(body, `a${t + 1}`, `${partial} + ${words[t]}`, LOOP);
    } else {
      next = out(destination, `a${t + 1}`, `${base} + ${hex(roundK(t))} + ${words[t]}`, level);
    }
    const rotated = target === tail
      ? out(tail, `c${t + 1}`, `rol(${b}, 30u)`, LOOP)
      : out(sink(levelOf(b)), `c${t + 1}`, `rol(${b}, 30u)`, levelOf(b));
    e = d;
    d = c;
    c = rotated;
    b = a;
    a = next;
  };
  for (let t = start; t < 76; t++) round(t);
  const a76 = a;
  for (let t = 76; t < 80; t++) round(t, tail);

  const code = `// Generated by sha1-shader.js for ${digits}-digit nonces.
@group(0) @binding(0) var<storage, read> rows: array<u32>;
struct Result { found: atomic<u32>, sample: array<u32, 5> };
@group(0) @binding(1) var<storage, read_write> result: Result;

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
  let offsetBase = rows[rb + ${SLOT.offset}u] + t * 10u - rows[rb + ${SLOT.jFirst}u];
  let expected = rows[rb + ${SLOT.a76}u];
  ${pre.join("\n  ")}
  for (var j = jFirst; j <= jLast; j = j + 1u) {
    ${body.join("\n    ")}
    let index = offsetBase + j;
    if (${a76} == expected || index == 0u) {
      ${tail.join("\n      ")}
      let h0 = ${hex(IV[0])} + ${a};
      let h1 = ${hex(IV[1])} + ${b};
      let h2 = ${hex(IV[2])} + ${c};
      let h3 = ${hex(IV[3])} + ${d};
      let h4 = ${hex(IV[4])} + ${e};
      if (index == 0u) {
        result.sample[0] = h0;
        result.sample[1] = h1;
        result.sample[2] = h2;
        result.sample[3] = h3;
        result.sample[4] = h4;
      }
      if (h0 == rows[rb + ${SLOT.target}u] && h1 == rows[rb + ${SLOT.target + 1}u] &&
          h2 == rows[rb + ${SLOT.target + 2}u] && h3 == rows[rb + ${SLOT.target + 3}u] &&
          h4 == rows[rb + ${SLOT.target + 4}u]) {
        atomicMin(&result.found, index);
      }
    }
  }
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
  record[SLOT.offset] = range.offset;
}
