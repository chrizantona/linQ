import test from 'node:test';
import assert from 'node:assert/strict';
import {renderCameras} from '../console-cameras.js';
import {renderAnalytics} from '../console-analytics.js';
const time=Date.parse('2026-10-24T12:00:00Z');
const phone={sample:{id:'phone',title:'<img src=x onerror=alert(1)>',case:'phone',image:'assets/phone.jpg',video:'assets/phone.mp4',source:'Source',sourceUrl:'javascript:alert(1)'},sourceId:'SRC-001',title:'Phone',case:'phone',timestamp:time,job:{status:'ready',analysis:{assessment:'unknown',latencyMs:200,analyzedAt:new Date(time).toISOString()},result:{frames:[{time:0,detections:[{label:'person'},{label:'phone'}]}],episode:{frames:['frame.jpg']}}},review:null};
const fall={sample:{id:'fall',title:'Fall',case:'fall',image:'assets/fall.jpg'},sourceId:'SRC-002',title:'Fall',case:'fall',timestamp:null,job:null,review:null};
test('camera page renders source jobs safely and catalog filters remain strict',()=>{
 const html=renderCameras({records:[phone,fall],selectedId:'fall',viewerMarkup:'<div>trusted viewer</div>',caseFilter:'phone'});
 assert.match(html,/data-camera-source="phone"/);assert.doesNotMatch(html,/data-camera-source="fall"/);
 assert.match(html,/&lt;img src=x onerror=alert\(1\)&gt;/);assert.doesNotMatch(html,/href="javascript:/);
 assert.match(html,/trusted viewer/);assert.doesNotMatch(html,/undefined|NaN/);
});
test('unprocessed source counts remain unknown and detected classes have Russian names',()=>{
 const empty=renderCameras({records:[fall],selectedId:'fall'});assert.match(empty,/<b>—<\/b>/);
 const analyzed=renderCameras({records:[phone],selectedId:'phone'});assert.match(analyzed,/Телефон/);assert.doesNotMatch(analyzed,/>phone<\/span>/);
});
test('analytics renders waiting and uncertain observations without nested list items or invented operator decisions',()=>{
 const html=renderAnalytics({records:[phone,fall],now:time+1000});
 assert.doesNotMatch(html,/<li>\s*<li>/);assert.doesNotMatch(html,/NaN|undefined/);
 assert.match(html,/Ожидают решения оператора: 1/);assert.match(html,/Неопределённых оценок модели: 1/);
 assert.match(html,/Оператор · подтвердил<\/span><strong>0<\/strong>/);
});

test('calendar trend labels the current day instead of the preceding rolling-day bucket',()=>{
 const html=renderAnalytics({records:[phone],now:time+1000,periodDays:7});
 assert.match(html,/title="24 окт\.: 1 записей"/);
});

test('camera view exposes actual early boxes counts and classes while scene analysis is pending',()=>{
 const row={...phone,job:{...phone.job,status:'analyzing',cv_ready:true,vlm_pending:true,analysis:null,result:{frames:[{time:0,detections:[{label:'person',trackId:0},{label:'phone',trackId:1}]}],episode:null}}};
 const html=renderCameras({records:[row],selectedId:'phone'});
 assert.match(html,/<b>1<\/b><span>людей в кадре/);assert.match(html,/<b>1<\/b><span>объектов/);assert.match(html,/camera-class-chip">Телефон/);
 assert.match(html,/Видео с рамками доступно/);assert.match(html,/<b>0<\/b> разобрано/);assert.doesNotMatch(html,/Подтверждено|Результат готов/);
});

test('camera and analytics agree that completed CV-only video is processed',()=>{
 const row={...phone,job:{...phone.job,status:'ready',analysis:null,result:{frames:[{time:0,detections:[{label:'person',trackId:0}]}],episode:null}}};
 const cameras=renderCameras({records:[row],selectedId:'phone'});
 const analytics=renderAnalytics({records:[row],now:time+1000});
 assert.match(cameras,/<b>1<\/b> разобрано/);assert.doesNotMatch(cameras,/Ожидает решения/);
 assert.doesNotMatch(analytics,/NaN|undefined/);assert.match(analytics,/Модель · неопределённо<\/span><strong>0<\/strong>/);
});
