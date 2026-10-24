export const CASE_NAMES = Object.freeze({
  fall: 'Падение',
  proximity: 'Опасное сближение',
  phone: 'Использование телефона',
  smoking: 'Курение',
});

export const CASE_COLORS = Object.freeze({
  fall: '#487bcc',
  proximity: '#db8051',
  phone: '#62a384',
  smoking: '#c97aa0',
});

const validMillis = value => {
  if (value == null || value === '') return null;
  const millis = typeof value === 'number' ? value : Date.parse(value);
  return Number.isFinite(millis) ? millis : null;
};
const timestampOf = item => validMillis(item?.analysis?.analyzedAt) ?? validMillis(item?.analyzedAt) ?? validMillis(item?.finishedAt) ?? validMillis(item?.createdAt) ?? null;
const newestFirst = (a, b) => (timestampOf(b) ?? -Infinity) - (timestampOf(a) ?? -Infinity);


export function buildRecords(samples = [], store = {}, { modelId = null } = {}) {
  const jobs = Array.isArray(store?.jobs) ? store.jobs : [];
  const reviews = Array.isArray(store?.reviews) ? store.reviews : [];
  const latestReview = new Map();
  for (const review of reviews) {
    if (!review?.jobId) continue;
    const prior = latestReview.get(review.jobId);
    if (!prior || (validMillis(review.reviewedAt) ?? -Infinity) >= (validMillis(prior.reviewedAt) ?? -Infinity)) {
      latestReview.set(review.jobId, review);
    }
  }

  return samples.map((sample, index) => {
    const job = jobs.filter(item => item?.sampleId === sample?.id && (modelId == null || (item.modelRequested ?? item.analysis?.modelRequested) === modelId))
      .sort(newestFirst)[0] ?? null;
    const review = job ? latestReview.get(job.id) ?? null : null;
    return {
      sample,
      job,
      review,
      sourceId: `SRC-${String(index + 1).padStart(3, '0')}`,
      case: sample?.case ?? null,
      title: sample?.title ?? sample?.shortLabel ?? sample?.id ?? `Sample ${index + 1}`,
      timestamp: job ? timestampOf(job) : null,
    };
  });
}

const finiteNonnegative = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const percentile = (values, p) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)];
};
const median = values => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const uniqueDetectionCount = job => {
  const frames = job?.result?.frames;
  if (!Array.isArray(frames)) return 0;
  return frames.reduce((sum, frame) => sum + (Array.isArray(frame?.detections) ? frame.detections.length : 0), 0);
};
const frameCountFor = job => {
  const count = job?.result?.episode?.frames?.length;
  if (Number.isFinite(count)) return count;
  const metrics = finiteNonnegative(job?.result?.metrics?.frames);
  return metrics === null ? 0 : metrics;
};


export function summarize(records = []) {
  const rows = Array.isArray(records) ? records : [];
  const processedCV = rows.filter(({ job }) => job?.status === 'ready' && Boolean(job.result || job.analysis));
  const modelAnalyzed = rows.filter(({ job }) => job?.status === 'ready' && job.analysis && ['confirmed', 'not_confirmed', 'unknown'].includes(job.analysis.assessment));
  const assessments = rows.map(({ job }) => job?.analysis?.assessment);
  const latencies = rows.map(({ job }) => finiteNonnegative(job?.analysis?.latencyMs)).filter(value => value !== null);
  const costs = rows.map(({ job }) => finiteNonnegative(job?.analysis?.reportedCostUsd)).filter(value => value !== null);
  const costUsd = costs.length ? costs.reduce((sum, value) => sum + value, 0) : null;

  return {
    total: rows.length,
    analyzed: processedCV.length,
    processedCV: processedCV.length,
    modelAnalyzed: modelAnalyzed.length,
    waiting: rows.filter(({ job, review }) => job?.status === 'ready' && job.analysis && ['confirmed', 'not_confirmed', 'unknown'].includes(job.analysis.assessment) && !review).length,
    confirmed: rows.filter(({ review }) => review?.verdict === 'confirmed').length,
    rejected: rows.filter(({ review }) => review?.verdict === 'rejected').length,
    uncertain: rows.filter(({ review }) => review?.verdict === 'uncertain').length,
    modelConfirmed: assessments.filter(value => value === 'confirmed').length,
    modelRejected: assessments.filter(value => value === 'not_confirmed').length,
    modelUnknown: assessments.filter(value => value === 'unknown').length,
    active: rows.filter(({ job }) => ['detecting', 'analyzing'].includes(job?.status)).length,
    failed: rows.filter(({ job }) => job?.status === 'failed' || Boolean(job?.analysisError)).length,
    medianLatencyMs: median(latencies),
    p95LatencyMs: percentile(latencies, 0.95),
    reviewPercent: rows.length ? (rows.filter(({ review }) => Boolean(review)).length / rows.length) * 100 : null,
    frameCount: rows.reduce((sum, { job }) => sum + frameCountFor(job), 0),
    objectCount: rows.reduce((sum, { job }) => sum + uniqueDetectionCount(job), 0),
    costUsd,
  };
}
