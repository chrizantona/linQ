export const objectNames={person:'Человек',forklift:'Погрузчик',ladder:'Лестница',stairs:'Ступени',pallet:'Паллета','suspended load':'Подвешенный груз',barrel:'Бочка',phone:'Телефон',cigarette:'Сигарета',cup:'Стакан',helmet:'Каска','safety vest':'Жилет',chair:'Стул',table:'Стол',book:'Книга',bottle:'Бутылка',bench:'Скамейка',box:'Коробка',tool:'Инструмент',machine:'Оборудование',scaffold:'Подмости',handrail:'Перила',vehicle:'Транспорт',door:'Дверь',laptop:'Ноутбук',monitor:'Монитор',bed:'Кровать',sofa:'Диван',clipboard:'Планшет с бумагами',paper:'Лист бумаги',newspaper:'Газета',plant:'Растение',shelf:'Стеллаж',cigar:'Сигара',pipe:'Трубка'};
const subjects={phone:['phone'],smoking:['cigarette','cigar','pipe','cup','bottle'],fall:['ladder','stairs','handrail','scaffold'],proximity:['forklift','vehicle','machine','suspended load']};
export const isRelated=(d,scenario)=>(subjects[scenario]||[]).includes(d.label);
export const objectKey=(d,index=0)=>`${d.label}:${d.trackId??index}`;
export const objectText=d=>`${objectNames[d.label]||d.label}${d.trackId!=null?' #'+String(d.trackId).split('-').at(-1):''}`;
export const objectColor=(d,scenario)=>d.label==='person'?'#80b4ff':isRelated(d,scenario)?'#ffbd77':'#e2e7ef';
export function videosForResult(result){
 const videos=Array.isArray(result?.videos)?result.videos:[];
 const seen=new Set();
 return videos.filter(video=>typeof video?.camera==='string'&&video.camera&&typeof video.url==='string'&&!seen.has(video.camera)&&(seen.add(video.camera),true));
}
export function episodesForResult(result){
 const seen=new Set();
 return [...(Array.isArray(result?.episodes)?result.episodes:[]),...(result?.episode?[result.episode]:[])].filter(episode=>{
  if(!episode?.eventId)return true;
  const key=`${episode.camera||result?.primaryCamera||''}\0${episode.eventId}`;
  if(seen.has(key))return false;
  seen.add(key);return true;
 });
}
export function framesForCamera(result,cameraId){
 const byCamera=result?.cameraFrames||result?.framesByCamera;
 if(byCamera&&typeof byCamera==='object'&&!Array.isArray(byCamera)){
  const frames=byCamera[cameraId];return Array.isArray(frames)?frames.filter(frame=>!frame?.camera||frame.camera===cameraId):[];
 }
 if(result?.primaryCamera&&cameraId&&result.primaryCamera!==cameraId)return [];
 const frames=Array.isArray(result?.frames)?result.frames:[];
 return frames.filter(frame=>!cameraId||!frame?.camera||frame.camera===cameraId);
}
export function framesForImageIndex(result,cameraId,imageIndex){
 return framesForCamera(result,cameraId).filter(frame=>frame?.image_index===imageIndex);
}
export function frameAtOrBefore(frames,time){
 return (Array.isArray(frames)?frames:[]).filter(frame=>Number.isFinite(frame?.time)&&frame.time<=time).reduce((latest,frame)=>!latest||frame.time>latest.time?frame:latest,null);
}
export function imageIndexForEvidence(evidence,current=0){
 const index=evidence?.image_index;
 return Number.isInteger(index)&&index>=0?index:current;
}
export function episodeForSelection(result,eventId,cameraId){
 const episodes=episodesForResult(result);
 return episodes.find(episode=>(!eventId||episode.eventId===eventId)&&(!cameraId||!episode.camera||episode.camera===cameraId))||null;
}
export function reviewForSelection(reviews,jobId,eventId=''){
 return (Array.isArray(reviews)?reviews:[]).findLast(review=>review?.jobId===jobId&&(eventId?review.eventId===eventId:!review.eventId))||null;
}
export const reviewDraftKey=(jobId,eventId='')=>`${jobId||''}\0${eventId||''}`;
export function reviewPayload({jobId,eventId='',verdict,note=''}){
 const payload={jobId,verdict,note};
 if(eventId)payload.eventId=eventId;
 return payload;
}
export function eventStatuses(event,task=''){
 const response=event?.vlm?.response,presence=['phone_objects','cigarette_objects'].includes(task),zone=eventTypeKey(event).startsWith('zone_');
 const candidate=event?.actionCandidate,insufficient=typeof candidate==='string'&&candidate.startsWith('insufficient');
 const candidateStatus=event?.actionCandidateStatus;
 const model=presence?'Модель: объектный сигнал':zone?'Модель: геометрический кандидат':candidate==='visible_action_candidate'?'Модель: кандидат события':candidate==='no_visible_action_candidate'?'Модель: не кандидат события':candidate==='unknown'||insufficient||['insufficient_object_evidence','model_response_unavailable'].includes(candidateStatus)?'Модель: недостаточно данных':candidate===false||candidate==='not_candidate'?'Модель: не кандидат события':!response||typeof response.unsafe!=='boolean'?'Модель: недостаточно данных':response.unsafe?'Модель: кандидат события':'Модель: не кандидат события';
 const review=event?.review,operator=review?review.confirmed?'Подтверждено оператором':'Отклонено оператором':'Ожидает решения оператора';
 return {model,operator};
}
export function eventJournalPresentation(event,task=''){
 const statuses=eventStatuses(event,task),presence=['phone_objects','cigarette_objects'].includes(task),zone=eventTypeKey(event).startsWith('zone_');
 return {...statuses,showVlmEvidence:!presence&&!zone,imageIndex:Number.isInteger(event?.image_index)&&event.image_index>=0?event.image_index:null};
}
export function actionEvidencePresentation(analysis){
 if(analysis?.actionCandidateStatus!=='insufficient_object_evidence')return null;
 return {status:'Недостаточно данных',primary:'В кадрах эпизода нет подтверждения от детектора предметов; действие не установлено.',observation:analysis.modelObservation||{reason:analysis.summary,evidence:analysis.facts||[]}};
}
export const eventTypeKey=event=>event?.rule_id||event?.kind||'';
export function eventTypeLabel(event,taskTitle=''){
 const key=typeof event==='string'?event:eventTypeKey(event);
 const labels={inspection:'Проверка сцены',phone_present:'Телефон обнаружен',cigarette_present:'Сигарета обнаружена',phone_use_review:'Использование телефона · кандидат',smoking_review:'Курение · кандидат'};
 if(labels[key])return labels[key];
 if(key.startsWith('zone_'))return 'Вход в зону';
 if(key.endsWith('_clip_candidate')&&taskTitle)return `${taskTitle} · кандидат`;
 return key||'Событие';
}
export function filterEventRecords(events,{camera='',type='',zone='',start='',end=''}={}){
 const lower=Number(start),upper=Number(end);
 return (Array.isArray(events)?events:[]).filter(event=>
  (!camera||event?.camera_id===camera)&&
  (!type||eventTypeKey(event)===type)&&
  (!zone||event?.zone_id===zone)&&
  (start===''||Number.isFinite(lower)&&Number(event?.trigger_timestamp_s)>=lower)&&
  (end===''||Number.isFinite(upper)&&Number(event?.trigger_timestamp_s)<=upper));
}
const overlap=(a,b)=>Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));
export function placeAnnotations(detections,{x=0,y=0,w,h},scenario,focused=''){
 const clamp=(n,a,b)=>Math.max(a,Math.min(n,b));
 const valid=(detections||[]).map((d,index)=>({d,index,key:objectKey(d,index)})).filter(({d})=>Array.isArray(d.box)&&d.box.length===4&&d.box.every(Number.isFinite)&&d.box[2]>d.box[0]&&d.box[3]>d.box[1]);
 const boxes=valid.map(o=>({...o,x:x+o.d.box[0]*w,y:y+o.d.box[1]*h,w:(o.d.box[2]-o.d.box[0])*w,h:(o.d.box[3]-o.d.box[1])*h}));
 boxes.sort((a,b)=>Number(b.key===focused)-Number(a.key===focused)||Number(isRelated(b.d,scenario))-Number(isRelated(a.d,scenario))||Number(b.d.label==='person')-Number(a.d.label==='person')||a.index-b.index);
 const tags=[];
 return boxes.map(box=>{
  const text=objectText(box.d),font=w<420?11:12,tw=Math.min(w-8,Math.max(86,text.length*font*.58+24)),th=25;
  const candidates=[[box.x,box.y-th-6],[box.x+box.w+7,box.y],[box.x-tw-7,box.y],[box.x,box.y+box.h+6],[box.x+box.w-tw,box.y-th-6],[x+4,y+4],[x+w-tw-4,y+4],[x+4,y+h-th-4],[x+w-tw-4,y+h-th-4]].map(([a,b])=>({x:clamp(a,x+4,x+w-tw-4),y:clamp(b,y+4,y+h-th-4),w:tw,h:th}));
  const score=tag=>boxes.reduce((sum,other)=>sum+overlap(tag,other)*(isRelated(other.d,scenario)?14:other.d.label==='person'?.2:.8),0)+tags.reduce((sum,other)=>sum+overlap(tag,other)*30,0)+Math.hypot(tag.x+tw/2-(box.x+box.w/2),tag.y+th/2-(box.y+box.h/2))*4;
  const tag=candidates.reduce((best,c)=>score(c)<score(best)?c:best,candidates[0]);tags.push(tag);
  const from=[clamp(tag.x+tw/2,box.x,box.x+box.w),clamp(tag.y+th/2,box.y,box.y+box.h)];
  const to=[clamp(from[0],tag.x,tag.x+tw),clamp(from[1],tag.y,tag.y+th)];
  return {...box,text,font,color:objectColor(box.d,scenario),related:isRelated(box.d,scenario),selected:box.key===focused,tag,leader:{from,to}};
 });
}
