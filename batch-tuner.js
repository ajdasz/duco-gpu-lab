export class BatchTuner {
  constructor(sizes, { minimumMs = 35, maximumMs = 65, targetMs = 50 } = {}) {
    this.sizes = [...sizes].sort((a, b) => a - b);
    this.minimumMs = minimumMs;
    this.maximumMs = maximumMs;
    this.targetMs = targetMs;
    this.reset();
  }

  reset() {
    this.currentSize = null;
    this.samples = [];
  }

  observe(size, elapsedMs) {
    const index = this.sizes.indexOf(size);
    if (index < 0 || !Number.isFinite(elapsedMs) || elapsedMs <= 0) return size;
    if (this.currentSize !== size) {
      this.currentSize = size;
      this.samples = [];
    }
    this.samples.push(elapsedMs);
    if (this.samples.length > 3) this.samples.shift();
    if (this.samples.length < 3) return size;

    const median = [...this.samples].sort((a, b) => a - b)[1];
    let next = size;
    if (median < this.minimumMs && index < this.sizes.length - 1) next = this.sizes[index + 1];
    if (median > this.maximumMs && index > 0) next = this.sizes[index - 1];
    if (next === size) return size;

    const projectedMs = median * next / size;
    if (Math.abs(projectedMs - this.targetMs) >= Math.abs(median - this.targetMs)) return size;
    this.reset();
    return next;
  }
}
