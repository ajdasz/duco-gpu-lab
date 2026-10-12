// Optimized DUCO-S1 for a 40-character ASCII previous hash.
// The ten fixed prefix words are packed once on the CPU for each job.
@group(0) @binding(0) var<storage, read> params: array<u32>;

struct Result {
  found: atomic<u32>,
  sample0: u32,
  sample1: u32,
  sample2: u32,
  sample3: u32,
  sample4: u32,
};
@group(0) @binding(1) var<storage, read_write> result: Result;

fn rol(value: u32, bits: u32) -> u32 {
  return (value << bits) | (value >> (32u - bits));
}

fn digit_count(value: u32) -> u32 {
  if (value >= 1000000000u) { return 10u; }
  if (value >= 100000000u) { return 9u; }
  if (value >= 10000000u) { return 8u; }
  if (value >= 1000000u) { return 7u; }
  if (value >= 100000u) { return 6u; }
  if (value >= 10000u) { return 5u; }
  if (value >= 1000u) { return 4u; }
  if (value >= 100u) { return 3u; }
  if (value >= 10u) { return 2u; }
  return 1u;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x + invocation.y * 8388480u;
  if (index >= params[17]) { return; }
  let low_sum = params[16] + index;
  let high = params[15] + low_sum / 10000000u;
  var low = low_sum % 10000000u;
  var digits = digit_count(low);
  if (high > 0u) { digits = digit_count(high) + 7u; }

  var words: array<u32, 16>;
  for (var i = 0u; i < 10u; i = i + 1u) { words[i] = params[i]; }
  for (var i = 10u; i < 16u; i = i + 1u) { words[i] = 0u; }

  var high_remaining = high;
  for (var i = digits; i > 0u; i = i - 1u) {
    let position = 40u + i - 1u;
    let word = position >> 2u;
    let shift = 24u - ((position & 3u) * 8u);
    var digit: u32;
    if (high > 0u && i <= digits - 7u) {
      digit = high_remaining % 10u;
      high_remaining = high_remaining / 10u;
    } else {
      digit = low % 10u;
      low = low / 10u;
    }
    words[word] = words[word] | ((48u + digit) << shift);
  }
  let padding_position = 40u + digits;
  words[padding_position >> 2u] = words[padding_position >> 2u] |
    (128u << (24u - ((padding_position & 3u) * 8u)));
  words[15] = (40u + digits) * 8u;

  // Rounds 0 to 9 depend only on the fixed 40-byte prefix.
  var a = params[18];
  var b = params[19];
  var c = params[20];
  var d = params[21];
  var e = params[22];

  for (var i = 10u; i < 16u; i = i + 1u) {
    let slot = i & 15u;
    let next = rol(a, 5u) + ((b & c) | ((~b) & d)) + e + 0x5a827999u + words[slot];
    e = d;
    d = c;
    c = rol(b, 30u);
    b = a;
    a = next;
  }
  for (var i = 16u; i < 20u; i = i + 1u) {
    let slot = i & 15u;
    words[slot] = rol(words[(i + 13u) & 15u] ^ words[(i + 8u) & 15u] ^
                      words[(i + 2u) & 15u] ^ words[slot], 1u);
    let next = rol(a, 5u) + ((b & c) | ((~b) & d)) + e + 0x5a827999u + words[slot];
    e = d;
    d = c;
    c = rol(b, 30u);
    b = a;
    a = next;
  }
  for (var i = 20u; i < 40u; i = i + 1u) {
    let slot = i & 15u;
    words[slot] = rol(words[(i + 13u) & 15u] ^ words[(i + 8u) & 15u] ^
                      words[(i + 2u) & 15u] ^ words[slot], 1u);
    let next = rol(a, 5u) + (b ^ c ^ d) + e + 0x6ed9eba1u + words[slot];
    e = d;
    d = c;
    c = rol(b, 30u);
    b = a;
    a = next;
  }
  for (var i = 40u; i < 60u; i = i + 1u) {
    let slot = i & 15u;
    words[slot] = rol(words[(i + 13u) & 15u] ^ words[(i + 8u) & 15u] ^
                      words[(i + 2u) & 15u] ^ words[slot], 1u);
    let next = rol(a, 5u) + ((b & c) | (b & d) | (c & d)) + e + 0x8f1bbcdcu + words[slot];
    e = d;
    d = c;
    c = rol(b, 30u);
    b = a;
    a = next;
  }
  for (var i = 60u; i < 80u; i = i + 1u) {
    let slot = i & 15u;
    words[slot] = rol(words[(i + 13u) & 15u] ^ words[(i + 8u) & 15u] ^
                      words[(i + 2u) & 15u] ^ words[slot], 1u);
    let next = rol(a, 5u) + (b ^ c ^ d) + e + 0xca62c1d6u + words[slot];
    e = d;
    d = c;
    c = rol(b, 30u);
    b = a;
    a = next;
  }

  let h0 = 0x67452301u + a;
  let h1 = 0xefcdab89u + b;
  let h2 = 0x98badcfeu + c;
  let h3 = 0x10325476u + d;
  let h4 = 0xc3d2e1f0u + e;
  if (index == 0u) {
    result.sample0 = h0;
    result.sample1 = h1;
    result.sample2 = h2;
    result.sample3 = h3;
    result.sample4 = h4;
  }
  if (h0 == params[10] && h1 == params[11] && h2 == params[12] &&
      h3 == params[13] && h4 == params[14]) {
    atomicMin(&result.found, index);
  }
}
