import test from 'node:test';
import assert from 'node:assert/strict';
import {placeAnnotations,objectNames,objectText,isRelated,videosForResult,framesForCamera,framesForImageIndex,frameAtOrBefore,episodesForResult,episodeForSelection,imageIndexForEvidence,eventStatuses,eventJournalPresentation,actionEvidencePresentation,eventTypeKey,eventTypeLabel,filterEventRecords,reviewForSelection,reviewDraftKey,reviewPayload} from '../viewer-annotations.js';
const detections=[{label:'person',trackId:'person-1',box:[.2,.05,.9,.95]},{label:'phone',trackId:'phone-1',box:[.55,.6,.68,.72]},{label:'bench',trackId:'bench-1',box:[.05,.7,.95,.98]}];
test('labels stay in source bounds and preserve all real objects',()=>{
 const raw=JSON.stringify(detections),annotations=placeAnnotations(detections,{x:20,y:15,w:640,h:360},'phone');
 assert.equal(annotations.length,3);assert.equal(annotations[0].d.label,'phone');
 for(const a of annotations){assert.ok(a.tag.x>=20&&a.tag.y>=15);assert.ok(a.tag.x+a.tag.w<=660&&a.tag.y+a.tag.h<=375);}
 const phone=annotations.find(a=>a.d.label==='phone');
 assert.ok(phone.tag.x+phone.tag.w<=phone.x||phone.tag.x>=phone.x+phone.w||phone.tag.y+phone.tag.h<=phone.y||phone.tag.y>=phone.y+phone.h,'label avoids the small action object');
 assert.equal(JSON.stringify(detections),raw);
});
test('same track label is selectable and context taxonomy is translated',()=>{
 const annotations=placeAnnotations(detections,{x:0,y:0,w:360,h:220},'phone','person:person-1');
 assert.equal(annotations[0].d.label,'person');assert.equal(annotations[0].selected,true);
 assert.equal(objectText(detections[1]),'Телефон #1');assert.equal(objectNames['safety vest'],'Жилет');assert.equal(objectNames.stairs,'Ступени');assert.equal(objectNames.clipboard,'Планшет с бумагами');assert.equal(objectNames.bed,'Кровать');assert.equal(objectNames.pipe,'Трубка');
 assert.equal(isRelated({label:'cup'},'smoking'),true);assert.equal(isRelated({label:'person'},'phone'),false);
});
test('malformed boxes are not turned into annotations',()=>{
 assert.equal(placeAnnotations([{label:'phone',box:[.5,.5,.2,.6]},{label:'person',box:[0,0,NaN,1]}],{w:360,h:220},'phone').length,0);
});

test('numeric and missing actual tracker IDs remain usable',()=>{
 assert.equal(objectText({label:'person',trackId:0}),'Человек #0');
 assert.equal(objectText({label:'person',trackId:12}),'Человек #12');
 assert.equal(objectText({label:'person',trackId:null}),'Человек');
 assert.equal(placeAnnotations([{label:'person',trackId:0,box:[.1,.1,.3,.8]}],{w:640,h:360},'fall')[0].key,'person:0');
});

test('camera selection never borrows detections from another camera',()=>{
 const result={primaryCamera:'cam-a',frames:[{camera:'cam-a',time:1,detections:[{label:'person'}]}],cameraFrames:{'cam-a':[{camera:'cam-a',time:1,detections:[{label:'person'}]}],'cam-b':[{camera:'cam-b',time:2,detections:[{label:'phone'}]}]},videos:[{camera:'cam-a',url:'/a.mp4'},{camera:'cam-b',url:'/b.mp4'},{camera:'cam-b',url:'/duplicate.mp4'}]};
 assert.deepEqual(videosForResult(result).map(item=>item.camera),['cam-a','cam-b']);
 assert.deepEqual(framesForCamera(result,'cam-b').map(frame=>frame.detections[0].label),['phone']);
 assert.deepEqual(framesForCamera({primaryCamera:'cam-a',frames:result.frames},'cam-b'),[]);
});

test('event navigation can select actual camera boxes at or before its trigger time',()=>{
 const frames=[{camera:'cam-a',time:2,detections:[{label:'person'}]},{camera:'cam-a',time:4,detections:[{label:'phone'}]},{camera:'cam-b',time:4,detections:[{label:'cigarette'}]}];
 assert.deepEqual(frameAtOrBefore(frames.filter(frame=>frame.camera==='cam-a'),4.2),frames[1]);
 assert.equal(frameAtOrBefore(frames.filter(frame=>frame.camera==='cam-a'),1),null);
});

test('still-image review shows detections only for the selected source image',()=>{
 const result={cameraFrames:{'cam-a':[{camera:'cam-a',image_index:0,detections:[{label:'phone'}]},{camera:'cam-a',image_index:1,detections:[{label:'person'}]}]}};
 assert.deepEqual(framesForImageIndex(result,'cam-a',0).map(frame=>frame.detections[0].label),['phone']);
 assert.deepEqual(framesForImageIndex(result,'cam-a',1).map(frame=>frame.detections[0].label),['person']);
 assert.deepEqual(framesForImageIndex(result,'cam-a',2),[]);
});

test('evidence navigation uses the source image index, never the episode list position',()=>{
 const episode={eventId:'episode-4',image_index:3};
 const result={cameraFrames:{'cam-a':[0,1,2,3].map(image_index=>({camera:'cam-a',image_index,detections:[{label:`object-${image_index}`}]}))}};
 const selected=imageIndexForEvidence(episode,1);
 assert.equal(selected,3);
 assert.deepEqual(framesForImageIndex(result,'cam-a',selected).map(frame=>frame.detections[0].label),['object-3']);
 assert.equal(imageIndexForEvidence({eventId:'legacy-episode'},2),2,'missing evidence metadata keeps the current source index instead of guessing from list position');
});

test('episode and journal filtering are scoped to selected event, camera, zone and source time',()=>{
 const result={episode:{eventId:'a',camera:'cam-a'},episodes:[{eventId:'a',camera:'cam-a'},{eventId:'b',camera:'cam-b'}]};
 assert.equal(episodeForSelection(result,'b','cam-b').eventId,'b');
 assert.equal(episodeForSelection(result,'a','cam-b'),null);
 const events=[{camera_id:'cam-a',kind:'inspection',rule_id:'fall',zone_id:'site-a',trigger_timestamp_s:2},{camera_id:'cam-b',kind:'phone',zone_id:null,trigger_timestamp_s:5}];
 assert.equal(eventTypeKey(events[0]),'fall','incident type follows rule_id even when kind is inspection');
 assert.deepEqual(filterEventRecords(events,{camera:'cam-a',type:'fall',zone:'site-a',start:'1',end:'3'}),[events[0]]);
 assert.deepEqual(filterEventRecords(events,{camera:'cam-b',start:'6'}),[]);
});

test('episode selector keeps every camera episode available',()=>{
 const result={episodes:[{eventId:'a',camera:'cam-a'},{eventId:'b',camera:'cam-b'}],episode:{eventId:'c',camera:'cam-c'}};
 assert.equal(episodesForResult(result).length,3);
 assert.deepEqual(episodesForResult(result).map(item=>item.eventId),['a','b','c']);
});

test('primary episode duplicates are removed while other-camera episodes remain',()=>{
 const result={primaryCamera:'cam-b',episodes:[{eventId:'a',camera:'cam-a'},{eventId:'b',camera:'cam-b'}],episode:{eventId:'b'}};
 assert.deepEqual(episodesForResult(result).map(item=>[item.camera||result.primaryCamera,item.eventId]),[['cam-a','a'],['cam-b','b']]);
});

test('main-card reviews and drafts follow the selected event within a job',()=>{
 const reviews=[{jobId:'job-1',eventId:'primary',verdict:'confirmed'},{jobId:'job-1',eventId:'secondary',verdict:'rejected'},{jobId:'job-2',eventId:'secondary',verdict:'uncertain'}];
 assert.equal(reviewForSelection(reviews,'job-1','secondary').verdict,'rejected');
 assert.equal(reviewForSelection(reviews,'job-1','missing'),null,'another event verdict must not leak into the selected card');
 assert.equal(reviewForSelection(reviews,'job-2','secondary').verdict,'uncertain');
 assert.notEqual(reviewDraftKey('job-1','primary'),reviewDraftKey('job-1','secondary'));
 assert.deepEqual(reviewPayload({jobId:'job-1',eventId:'secondary',verdict:'rejected',note:'checked'}),{jobId:'job-1',eventId:'secondary',verdict:'rejected',note:'checked'});
 assert.deepEqual(reviewPayload({jobId:'job-1',verdict:'uncertain',note:''}),{jobId:'job-1',verdict:'uncertain',note:''});
});

test('operator review remains separate from the model event status',()=>{
 const candidate={vlm:{response:{unsafe:true}}};
 const before=eventStatuses(candidate,'phone_use');
 const after=eventStatuses({...candidate,review:{confirmed:false,note:'reviewed'}},'phone_use');
 assert.equal(before.model,'Модель: кандидат события');
 assert.equal(after.model,before.model);
 assert.equal(before.operator,'Ожидает решения оператора');
 assert.equal(after.operator,'Отклонено оператором');
 assert.equal(eventStatuses({vlm:{response:{unsafe:true}}},'phone_objects').model,'Модель: объектный сигнал');
});

test('journal presentation keeps presence evidence and source-image index explicit',()=>{
 const event={image_index:4,vlm:{response:{unsafe:true,reason:'object facts'}}};
 const presence=eventJournalPresentation(event,'cigarette_objects');
 assert.equal(presence.model,'Модель: объектный сигнал');
 assert.equal(presence.imageIndex,4);
 assert.equal(presence.showVlmEvidence,false);
 const action=eventJournalPresentation(event,'smoking');
 assert.equal(action.model,'Модель: кандидат события');
 assert.equal(action.showVlmEvidence,true);
});

test('event type labels hide internal identifiers without changing filter keys',()=>{
 assert.equal(eventTypeLabel('inspection'),'Проверка сцены');
 assert.equal(eventTypeLabel('phone_present'),'Телефон обнаружен');
 assert.equal(eventTypeLabel('cigarette_present'),'Сигарета обнаружена');
 assert.equal(eventTypeLabel('phone_use_review'),'Использование телефона · кандидат');
 assert.equal(eventTypeLabel('smoking_review'),'Курение · кандидат');
 assert.equal(eventTypeLabel('zone_pexels-phone-office'),'Вход в зону');
 assert.equal(eventTypeLabel('omnifall_v2_clip_candidate','Падение'),'Падение · кандидат');
 assert.equal(eventTypeKey({kind:'inspection',rule_id:'fall'}),'fall');
});

test('journal respects action-candidate evidence gating while retaining the raw observation',()=>{
 const event={actionCandidate:'unknown',actionCandidateStatus:'insufficient_object_evidence',image_index:1,modelObservation:{reason:'A visible person is holding a phone'},vlm:{response:{unsafe:true}}};
 const presentation=eventJournalPresentation(event,'phone_use');
 assert.equal(presentation.model,'Модель: недостаточно данных');
 assert.equal(presentation.showVlmEvidence,true);
 assert.equal(presentation.imageIndex,1);
 assert.equal(event.modelObservation.reason,'A visible person is holding a phone');
 assert.equal(eventStatuses({...event,actionCandidate:'visible_action_candidate'},'phone_use').model,'Модель: кандидат события');
 assert.equal(eventStatuses({...event,actionCandidate:'no_visible_action_candidate'},'phone_use').model,'Модель: не кандидат события');
});

test('zone entries stay geometric candidates and do not inherit phone-action observations',()=>{
 const event={rule_id:'zone_pexels-phone-office',actionCandidate:'visible_action_candidate',actionCandidateStatus:'review_candidate',vlm:{response:{unsafe:true,reason:'person is using a phone'}}};
 const presentation=eventJournalPresentation(event,'phone_use');
 assert.equal(presentation.model,'Модель: геометрический кандидат');
 assert.equal(presentation.operator,'Ожидает решения оператора');
 assert.equal(presentation.showVlmEvidence,false);
});

test('insufficient object evidence leads with an explicit unknown action outcome',()=>{
 const finding=actionEvidencePresentation({actionCandidateStatus:'insufficient_object_evidence',summary:'raw model output',facts:['raw fact'],modelObservation:{reason:'raw model output',evidence:['raw fact']}});
 assert.equal(finding.status,'Недостаточно данных');
 assert.equal(finding.primary,'В кадрах эпизода нет подтверждения от детектора предметов; действие не установлено.');
 assert.deepEqual(finding.observation,{reason:'raw model output',evidence:['raw fact']});
 assert.equal(actionEvidencePresentation({actionCandidateStatus:'review_candidate'}),null);
});
