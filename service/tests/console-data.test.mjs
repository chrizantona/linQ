import test from 'node:test';
import assert from 'node:assert/strict';
import { buildRecords, summarize } from '../console-data.js';

const samples = [
  {id:'s1',case:'fall',title:'Падение'},
  {id:'s2',case:'phone',title:'Телефон'},
  {id:'s3',case:'smoking',title:'Курение'},
];
const ready = (id, createdAt, assessment, extra={}) => ({id,sampleId:'s1',modelRequested:'linq-vision',status:'ready',createdAt,analysis:{assessment,...extra}});

test('buildRecords keeps catalog order, chooses latest requested-model job and latest review', () => {
  const store = {
    jobs:[
      ready('job-new','2026-10-23T12:00:00Z','not_confirmed',{analyzedAt:'2026-10-23T12:01:00Z'}),
      {...ready('other-model','2026-10-24T12:00:00Z','confirmed'),modelRequested:'openai/gpt-5.4-mini'},
      ready('job-old','2026-10-22T12:00:00Z','confirmed'),
    ],
    reviews:[
      {id:'r2',jobId:'job-new',verdict:'rejected',reviewedAt:'2026-10-23T13:00:00Z'},
      {id:'r1',jobId:'job-new',verdict:'confirmed',reviewedAt:'2026-10-23T12:30:00Z'},
    ],
  };
  const records = buildRecords(samples,store,{modelId:'linq-vision'});
  assert.deepEqual(records.map(row=>row.sourceId),['SRC-001','SRC-002','SRC-003']);
  assert.equal(records[0].job.id,'job-new');
  assert.equal(records[0].review.verdict,'rejected');
  assert.equal(records[0].timestamp,Date.parse('2026-10-23T12:01:00Z'));
  assert.equal(records[1].job,null);
  assert.equal(records[1].timestamp,null);
  assert.equal(records[2].sample,samples[2]);
});

test('summarize separates human decisions from model assessments and ignores unprocessed catalog rows', () => {
  const records = [
    {sample:samples[0],job:{status:'ready',analysis:{assessment:'confirmed',latencyMs:100,reportedCostUsd:0.02},result:{episode:{frames:['a','b']},frames:[{detections:[{},{}]}]}},review:{verdict:'rejected'}},
    {sample:samples[1],job:{status:'ready',analysis:{assessment:'unknown',latencyMs:300},result:{episode:{frames:['a']},frames:[{detections:[{}]}]}},review:null},
    {sample:samples[2],job:null,review:null},
    {sample:{id:'failed'},job:{status:'failed'},review:null},
  ];
  const stats = summarize(records);
  assert.equal(stats.total,4);
  assert.equal(stats.analyzed,2);
  assert.equal(stats.waiting,1);
  assert.equal(stats.confirmed,0);
  assert.equal(stats.rejected,1);
  assert.equal(stats.uncertain,0);
  assert.equal(stats.modelConfirmed,1);
  assert.equal(stats.modelRejected,0);
  assert.equal(stats.modelUnknown,1);
  assert.equal(stats.failed,1);
  assert.equal(stats.medianLatencyMs,200);
  assert.equal(stats.p95LatencyMs,300);
  assert.equal(stats.reviewPercent,25);
  assert.equal(stats.frameCount,3);
  assert.equal(stats.objectCount,3);
  assert.equal(stats.costUsd,0.02);
});

test('invalid timestamps and absent costs or latencies stay absent', () => {
  const records = buildRecords([{id:'sample',case:'fall',title:'Тест'}],{
    jobs:[{id:'bad-date',sampleId:'sample',modelRequested:'linq-vision',status:'ready',createdAt:'not-a-date',analysis:{analyzedAt:'also-invalid',assessment:'unknown'}}],
    reviews:[],
  });
  assert.equal(records[0].timestamp,null);
  const stats = summarize(records);
  assert.equal(stats.total,1);
  assert.equal(stats.analyzed,1);
  assert.equal(stats.modelUnknown,1);
  assert.equal(stats.medianLatencyMs,null);
  assert.equal(stats.p95LatencyMs,null);
  assert.equal(stats.costUsd,null);
  assert.equal(summarize([]).medianLatencyMs,null);
  assert.equal(summarize([]).costUsd,null);
});

test('LinQ jobs with no provider-specific model field retain partial boxes before analysis',()=>{
 const row=buildRecords([{id:'upload_1',case:'fall'}],{jobs:[{id:'job1',sampleId:'upload_1',status:'analyzing',createdAt:'2026-10-24T15:00:00Z',cv_ready:true,vlm_pending:true,result:{frames:[{time:0,detections:[{label:'person',trackId:0,box:[0,.1,.2,.9]}]}]},analysis:null}],reviews:[]})[0];
 assert.equal(row.job.id,'job1');assert.equal(row.job.result.frames[0].detections[0].trackId,0);assert.equal(row.job.analysis,null);
 const stats=summarize([row]);assert.equal(stats.active,1);assert.equal(stats.analyzed,0);assert.equal(stats.modelConfirmed,0);assert.equal(stats.waiting,0);
});

test('completed CV-only sources count as processed without an invented model verdict',()=>{
 const stats=summarize([{sample:{id:'cv-only'},job:{status:'ready',analysis:null,result:{frames:[{time:0,detections:[{label:'person',trackId:0}]}],metrics:{frames:1},episode:null}},review:null}]);
 assert.equal(stats.analyzed,1);assert.equal(stats.processedCV,1);assert.equal(stats.modelAnalyzed,0);assert.equal(stats.waiting,0);
 assert.equal(stats.modelConfirmed,0);assert.equal(stats.modelRejected,0);assert.equal(stats.modelUnknown,0);
});
