import {icon} from './icons.js';
import {placeAnnotations,videosForResult,framesForCamera,framesForImageIndex,frameAtOrBefore,episodesForResult,episodeForSelection,imageIndexForEvidence,eventJournalPresentation,actionEvidencePresentation,eventTypeKey,eventTypeLabel,filterEventRecords,reviewForSelection,reviewDraftKey,reviewPayload} from './viewer-annotations.js';
import {buildRecords,summarize,CASE_COLORS} from './console-data.js';
import {renderAnalytics} from './console-analytics.js';
import {renderCameras} from './console-cameras.js';
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pages=[['monitor','bell','Инциденты'],['events','bell','Журнал событий'],['analytics','grid','Аналитика'],['cameras','camera','Камеры']];
let queueFilter='all',periodDays=7,matrixMode='count',cameraFilter='all',cameraSort='catalog';
let samples=[],store={jobs:[],reviews:[],zones:{}},service={},selected='proximity-01',selectedModel='',kind='video',caseFilter='all',editing=false,polygon=[],boxes=true,pending=false,animationId=0,frameView=null,selectedJobId='',selectedCameraId='',selectedEventId='',selectedImageIndex=0,focusedObject='',viewerExpanded=false,uploadFiles=[],uploadKind='video',uploadTasks=[],uploadTask='synthsite',uploadUseVlm=null,uploadError='',refreshing=false,events=[],eventOptions=[],eventFilter={camera:'',type:'',zone:'',start:'',end:''},eventError='';
const reviewDrafts=new Map();
const eventReviewDrafts=new Map();
const caseNames={fall:'Падение',phone:'Телефон',smoking:'Курение',proximity:'Техника'};
const route=()=>pages.some(p=>p[0]===location.hash.slice(1))?location.hash.slice(1):'monitor';
const sample=()=>samples.find(s=>s.id===selected)||samples[0];
const requestedModel=j=>j?.modelRequested||j?.analysis?.modelRequested||j?.analysis?.model||service.model;
const latest=s=>s&&(store.jobs.find(j=>j.id===selectedJobId&&j.sampleId===s.id)||store.jobs.find(j=>j.sampleId===s.id));
const jobModel=j=>j?.analysis?.model||requestedModel(j);
const isPresenceTask=task=>['phone_objects','cigarette_objects'].includes(task);
function selectedEpisode(j){return episodeForSelection(j?.result,selectedEventId,selectedCameraId||j?.result?.primaryCamera||'');}
function cameraVideos(j){return videosForResult(j?.result);}
function cameraIdFor(j){const videos=cameraVideos(j),selectedEvent=events.find(event=>event.event_id===selectedEventId);return selectedCameraId&&(videos.some(video=>video.camera===selectedCameraId)||selectedEvent?.camera_id===selectedCameraId)?selectedCameraId:(j?.result?.primaryCamera||videos[0]?.camera||'');}
function cameraVideo(j){return cameraVideos(j).find(video=>video.camera===cameraIdFor(j))||null;}
function cameraSource(s,j){const video=cameraVideo(j);if(video)return video.sourceUrl||(video.camera===j?.result?.primaryCamera?s.video:video.url);return selectedCameraId&&selectedCameraId!==j?.result?.primaryCamera?null:s.video;}
function sampledFrame(j,time=0,preferRaw=false){
 const ep=selectedEpisode(j),cameraId=cameraIdFor(j),frame=frameAtOrBefore(framesForCamera(j?.result,cameraId),time);
 if(preferRaw&&frame)return {jobId:j.id,sampleId:j.sampleId,cameraId,eventId:selectedEventId||ep?.eventId||'',time:Number(frame.time),path:null};
 if(ep?.frames?.length&&Array.isArray(ep.frameTimes)&&ep.frameTimes.length){const index=ep.frameTimes.reduce((best,t,i)=>Math.abs(t-time)<Math.abs(ep.frameTimes[best]-time)?i:best,0);return {jobId:j.id,sampleId:j.sampleId,cameraId,eventId:ep.eventId,time:Number(ep.frameTimes[index]),path:ep.frames[index]};}
 return frame?{jobId:j.id,sampleId:j.sampleId,cameraId,eventId:selectedEventId||ep?.eventId||'',time:Number(frame.time),path:null}:null;
}
const activeFrame=j=>frameView?.sampleId===selected&&frameView?.jobId===j?.id&&(!frameView.cameraId||frameView.cameraId===cameraIdFor(j))&&(!frameView.eventId||frameView.eventId===selectedEventId||frameView.eventId===selectedEpisode(j)?.eventId)?frameView:null;
const duration=s=>Number(s?.duration)||0;
const imageSet=s=>s?.inputKind==='image_set'||Array.isArray(s?.sourceImages)&&s.sourceImages.length>1;
const imageSources=s=>Array.isArray(s?.sourceImages)?s.sourceImages:[];
const isStillImage=file=>['image/jpeg','image/png'].includes(file?.type)||/\.(jpe?g|png)$/i.test(file?.name||'');
const clock=t=>`${Math.floor(t/60).toString().padStart(2,'0')}:${(t%60).toFixed(2).padStart(5,'0')}`;
const safeLink=v=>/^https?:\/\//i.test(String(v||''))?esc(v):'#';
const review=j=>j&&store.reviews.findLast(r=>r.jobId===j.id);
const vlmPending=j=>j?.vlm_pending===true||j?.stage==='analyzing'||(j?.status==='analyzing'&&j?.vlm_pending!==false);
const processing=j=>j&&['detecting','analyzing','running','queued','rendering','collecting'].includes(j.status);
const label=j=>!j?'Ожидает запуска':j.status==='failed'?'Ошибка обработки':vlmPending(j)&&j.result?.frames?.length?'Анализируем событие':processing(j)?j.cv_ready?'Готовим результат':'Обработка видео':review(j)?'Проверено':j.result?.episode?'Ждёт решения':'Событий не выделено';
const badge=j=>`<span class="badge ${j?.status==='failed'?'bad':review(j)?'good':j?.result?.episode?'warn':''}"><i></i>${label(j)}</span>`;
async function api(path,body) {
 const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
 const data=await r.json();if(!r.ok)throw new Error(data.error?.message||(typeof data.detail==='string'?data.detail:null)||(typeof data.error==='string'?data.error:null)||'Не удалось выполнить запрос');return data;
}
function toast(text){$('#toast').textContent=text;$('#toast').classList.add('show');setTimeout(()=>$('#toast').classList.remove('show'),4000);}
const evidence=p=>esc(String(p).replace(/^\.runtime\/evidence\//,'/evidence/'));

const records=()=>buildRecords(samples,store);
const analyzed=j=>j?.analysis&&j.status==='ready';
const caseTitle=s=>({fall:'Падение',phone:'Использование телефона',smoking:'Курение',proximity:'Сближение с техникой'})[s?.case]||'Наблюдение';
const assessmentLabel=j=>!j?'Ожидает разбора':!analyzed(j)?label(j):isPresenceTask(j.task)?j.analysis.assessment==='confirmed'?'Объект обнаружен':j.analysis.assessment==='not_confirmed'?'Объект не обнаружен':'Нужна проверка':j.analysis.assessment==='confirmed'?'Кандидат события':j.analysis.assessment==='not_confirmed'?'Не подтверждено':'Нужна проверка';
const reviewLabel=r=>r?({confirmed:'Подтверждено оператором',rejected:'Ложный сигнал',uncertain:'Недостаточно данных'})[r.verdict]:'Ждёт решения';
const countText=(n,one,few,many)=>`${n} ${n%10===1&&n%100!==11?one:n%10>=2&&n%10<=4&&!(n%100>=12&&n%100<=14)?few:many}`;
const stamp=t=>Number.isFinite(t)?new Date(t).toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit'}):'—';
const ago=t=>!Number.isFinite(t)?'':Math.max(0,Date.now()-t)<60000?'сейчас':Math.max(0,Date.now()-t)<3600000?`${Math.floor((Date.now()-t)/60000)} мин`:`${Math.floor((Date.now()-t)/3600000)} ч`;
const byId=id=>records().find(r=>r.sample.id===id);
const eventLabel=(event,task='')=>eventTypeLabel(event,taskTitle(task));
const eventReview=event=>event?.review||null;
const taskProfile=id=>uploadTasks.find(task=>task.id===id)||null;
const taskTitle=id=>taskProfile(id)?.title_ru||id||'Профиль не указан';
const taskForCase=kind=>({fall:'omnifall_v2',proximity:'physicalai_v2',phone:'phone_use',smoking:'smoking'})[kind]||kind;
function eventList(){
 const jobs=new Map(store.jobs.map(job=>[job.id,job]));
 const records=filterEventRecords(events,eventFilter);
 return records.map(event=>{const job=jobs.get(event.job_id),sampleRecord=job&&byId(job.sampleId);return {event,job,sample:sampleRecord?.sample||samples.find(item=>item.id===event.sample_id)||null};});
}
function eventJournal(){
 const cameraValues=[...new Set(eventOptions.map(event=>event.camera_id).filter(Boolean))].sort();
 const typeValues=[...new Set(eventOptions.map(event=>eventTypeKey(event)).filter(Boolean))].sort();
 const zoneValues=[...new Set([...Object.keys(store.zones||{}),...eventOptions.map(event=>event.zone_id).filter(Boolean)])];
 const rows=eventList();
 const option=(value,label,current)=>`<option value="${esc(value)}" ${value===current?'selected':''}>${esc(label)}</option>`;
 return `<section class="event-journal"><div class="analytics-heading"><div><h1>Журнал событий</h1><p>${rows.length} кандидатов · решение модели и решение оператора показаны раздельно</p></div></div><form class="event-filters" id="event-filters" aria-label="Фильтры журнала"><label>Камера<select id="filter-camera">${option('','Все камеры',eventFilter.camera)}${cameraValues.map(value=>option(value,value.replace(/^user_camera_/,'Камера '),eventFilter.camera)).join('')}</select></label><label>Тип<select id="filter-type">${option('','Все типы',eventFilter.type)}${typeValues.map(value=>{const typeEvent=eventOptions.find(event=>eventTypeKey(event)===value),job=store.jobs.find(item=>item.id===typeEvent?.job_id);return option(value,eventLabel(typeEvent,job?.task),eventFilter.type);}).join('')}</select></label><label>Зона<select id="filter-zone">${option('','Все зоны',eventFilter.zone)}${zoneValues.map(value=>{const source=samples.find(item=>item.id===value);return option(value,source?.title||value,eventFilter.zone);}).join('')}</select></label><label>С начала, с<input id="filter-start" type="number" min="0" step="0.1" value="${esc(eventFilter.start)}" inputmode="decimal"></label><label>По, с<input id="filter-end" type="number" min="0" step="0.1" value="${esc(eventFilter.end)}" inputmode="decimal"></label><button class="btn" type="submit">Применить</button></form>${eventError?`<p class="event-error" role="alert">${esc(eventError)}</p>`:''}<div class="event-list" aria-live="polite">${rows.length?rows.map(({event,job,sample})=>{const review=eventReview(event),response=event?.modelObservation||event?.vlm?.response,presentation=eventJournalPresentation(event,job?.task),modelStatus=presentation.model,operatorStatus=presentation.operator,label=eventLabel(event,job?.task),profile=taskTitle(job?.task),imageEvent=job?.inputKind==='image_set'||job?.result?.inputKind==='image_set',ongoing=processing(job);return `<article class="event-row"><div class="event-row-main"><div><h2>${esc(label)}</h2><p>${esc(sample?.title||job?.sampleId||event.sample_id||'Источник')} · ${esc(event.camera_id?.replace(/^user_camera_/,'Камера ')||'Камера не указана')} · ${imageEvent?`Стоп-кадр ${Number(event.image_index??0)+1}`:`${Number(event.trigger_timestamp_s||0).toFixed(2)} с`}</p><small>${esc(profile)}</small></div><div class="event-statuses"><span class="status-pill">${esc(modelStatus)}</span>${ongoing?'<span class="status-pill attention">Обработка продолжается</span>':''}<span class="status-pill ${review?'':'attention'}">${esc(operatorStatus)}</span></div></div>${presentation.showVlmEvidence&&response?.reason?`<p class="event-reason"><strong>${event?.actionCandidateStatus==='insufficient_object_evidence'?'Неподтверждённое наблюдение модели':'Наблюдение модели'}:</strong> ${esc(response.reason)}</p>`:''}${presentation.showVlmEvidence&&response?.evidence?.length?`<ul class="event-facts">${response.evidence.map(fact=>`<li>${esc(fact)}</li>`).join('')}</ul>`:''}<div class="event-row-actions"><button class="btn quiet" type="button" data-event-open="${esc(event.event_id)}" data-job="${esc(event.job_id)}" ${!sample?'disabled':''}>Открыть материал и доказательства</button><label class="event-note-label">Комментарий<textarea data-event-note="${esc(event.event_id)}" maxlength="4000" placeholder="Контекст для решения оператора">${esc(eventReviewDrafts.get(event.event_id)??review?.note??'')}</textarea></label><div class="review-actions"><button class="btn primary" type="button" data-event-review="true" data-event-id="${esc(event.event_id)}" data-job="${esc(event.job_id)}" ${pending?'disabled':''}>Подтвердить</button><button class="btn pass" type="button" data-event-review="false" data-event-id="${esc(event.event_id)}" data-job="${esc(event.job_id)}" ${pending?'disabled':''}>Отклонить</button></div></div></article>`;}).join(''):'<div class="card empty-state compact"><h2>Событий пока нет</h2><p>После обработки видео найденные кандидаты появятся здесь.</p></div>'}</div></section>`;
}
function updateDom(current,next){
 for(const attribute of [...current.attributes])if(!next.hasAttribute(attribute.name)&&current.id!=='app')current.removeAttribute(attribute.name);
 for(const attribute of [...next.attributes])if(current.getAttribute(attribute.name)!==attribute.value)current.setAttribute(attribute.name,attribute.value);
 for(let index=0;index<next.childNodes.length;index++){
  const fresh=next.childNodes[index],old=current.childNodes[index];
  if(!old){current.append(fresh.cloneNode(true));continue;}
  if(old.nodeType!==fresh.nodeType||(old.nodeType===1&&(old.tagName!==fresh.tagName||old.id!==fresh.id))){old.replaceWith(fresh.cloneNode(true));continue;}
  if(old.nodeType!==1){if(old.textContent!==fresh.textContent)old.textContent=fresh.textContent;continue;}
  if(old.id==='media'&&old.tagName==='VIDEO'&&old.getAttribute('src')===fresh.getAttribute('src'))continue;
  const keepValue=old===document.activeElement&&['TEXTAREA','INPUT'].includes(old.tagName)&&old.type!=='checkbox';
  const value=keepValue?old.value:null;
  updateDom(old,fresh);
  if(keepValue)old.value=value;
  if(old.tagName==='INPUT'&&old.type==='checkbox')old.checked=fresh.checked;
 }
 while(current.childNodes.length>next.childNodes.length)current.lastChild.remove();
}
const boundMedia=new WeakSet();
function shell(content){
 const page=route(),stats=summarize(records()),videos=samples.filter(s=>s.video).length;
 document.title=`LinQ Safety — ${pages.find(p=>p[0]===page)[2]}`;
 const markup=`<header class="console-header"><div class="header-inner"><a class="console-brand" href="#monitor" aria-label="LinQ Safety — инциденты"><span class="brand-symbol">${icon('shield',18)}</span><span><strong>LinQ Safety</strong><small>${videos} видеозаписей · ${samples.length-videos} изображений и наборов кадров</small></span></a><nav class="console-nav" aria-label="Разделы">${pages.map(([id,i,title])=>`<a href="#${id}" ${page===id?'aria-current="page"':''}>${title}</a>`).join('')}</nav><div class="header-status"><span class="ready-indicator"><i></i>${service.linq_available===false?'Нет связи':store.jobs.some(processing)||pending?'Обработка':'Готов к работе'}</span><span class="demo-tag">Демо · источники</span><span class="operator-avatar" aria-label="Оператор">ОП</span></div></div></header><main id="main" class="console-main ${page==='monitor'?'incidents-main':''}">${content}</main><footer class="console-footer"><span>LinQ Safety · консоль оператора</span><span>Решения оператора сохраняются отдельно от вывода модели.</span></footer><div class="viewer-backdrop" hidden></div>`;
 const next=document.createElement('div');next.innerHTML=markup;updateDom($('#app'),next);
}
function queue(){
 const all=records(),pool=queueFilter==='reviewed'?all.filter(r=>r.review):all.filter(r=>!r.review);
 const filtered=pool.filter(r=>!['fall','phone','smoking','proximity'].includes(queueFilter)||r.case===queueFilter).filter(r=>queueFilter!=='unknown'||r.job?.analysis?.assessment==='unknown');
 const priority=s=>s.id==='phone-extra-01'?0:/extra|control-02|fall-lab/.test(s.id)?1:2;
 filtered.sort((a,b)=>priority(a.sample)-priority(b.sample));
 const waiting=all.filter(r=>!r.review&&analyzed(r.job)).length;
 const filters=[['all','Все',all.filter(r=>!r.review).length],['fall','Падения',all.filter(r=>!r.review&&r.case==='fall').length],['phone','Телефон',all.filter(r=>!r.review&&r.case==='phone').length],['smoking','Курение',all.filter(r=>!r.review&&r.case==='smoking').length],['proximity','Техника',all.filter(r=>!r.review&&r.case==='proximity').length],['unknown','Проверка',all.filter(r=>!r.review&&r.job?.analysis?.assessment==='unknown').length],['reviewed','Решены',all.filter(r=>r.review).length]];
 return `<aside class="card queue-panel source-panel" aria-label="Очередь инцидентов"><div class="queue-head"><div><h2>Очередь</h2><span>${waiting} ждут решения</span></div><div class="queue-filters" role="group" aria-label="Фильтр очереди">${filters.map(([id,title,count])=>`<button class="chip-filter ${queueFilter===id?'active':''}" data-queue-filter="${id}" aria-pressed="${queueFilter===id}">${title} <span>${count}</span></button>`).join('')}</div></div><div class="queue-list sources">${filtered.length?filtered.map(r=>`<button class="q-row ${selected===r.sample.id?'selected':''}" data-source="${esc(r.sample.id)}" aria-pressed="${selected===r.sample.id}" style="--case-color:${CASE_COLORS[r.case]||'#487bcc'}"><span class="queue-color"></span><span class="queue-copy"><span class="queue-title"><strong>${esc(caseTitle(r.sample))}</strong><time>${esc(ago(r.timestamp))}</time></span><span class="queue-source">${esc(r.sourceId)} · ${esc(r.title)}</span><span class="queue-tags"><span class="queue-result ${r.job?.analysis?.assessment==='unknown'?'attention':''}">${icon(r.job?.analysis?.assessment==='confirmed'?'check':'eye',12)}${esc(!r.sample.video?'Только просмотр':r.review?'Решение сохранено':assessmentLabel(r.job))}</span><span class="mini-tag">${r.sample.video?duration(r.sample).toFixed(1)+' с':imageSet(r.sample)?'Набор стоп-кадров':'Изображение'}</span></span></span></button>`).join(''):'<div class="empty-state compact">В этом фильтре пока нет эпизодов.</div>'}</div><a class="queue-library" href="#cameras">${icon('camera',16)}Все видео и изображения ${icon('arrow',14)}</a></aside>`;
}
function summary(){
 const rs=records(),s=summarize(rs),recent=rs.filter(r=>r.review).sort((a,b)=>new Date(b.review.reviewedAt||0)-new Date(a.review.reviewedAt||0)).slice(0,4);
 return `<aside class="session-summary" aria-label="Сводка сессии"><section class="card summary-card"><h2>Сессия</h2><div class="session-metrics"><div><strong>${s.waiting}</strong><span>Ждут решения</span></div><div><strong>${s.confirmed}</strong><span>Подтверждено</span></div><div><strong>${s.rejected}</strong><span>Ложных сигналов</span></div><div><strong>${s.uncertain}</strong><span>Нужна проверка</span></div></div><div class="progress-label"><span>Проверено оператором</span><b>${s.reviewPercent==null?'—':Number(s.reviewPercent.toFixed(1))+'%'}</b></div><div class="progress-bar"><span style="width:${s.reviewPercent||0}%"></span></div></section><section class="card summary-card"><div class="card-title"><h2>Разбор видео</h2></div><dl class="detail-metrics"><div><dt>В обработке</dt><dd>${s.active}</dd></div><div><dt>Разобрано источников</dt><dd>${s.analyzed}</dd></div></dl></section><section class="card summary-card"><h2>Выводы модели</h2><dl class="detail-metrics"><div><dt>Действие подтверждено</dt><dd>${s.modelConfirmed}</dd></div><div><dt>Не подтверждено</dt><dd>${s.modelRejected}</dd></div><div><dt>Недостаточно данных</dt><dd>${s.modelUnknown}</dd></div></dl><p class="muted-note">Окончательное решение принимает оператор.</p></section><section class="card summary-card"><h2>Последние решения</h2>${recent.length?`<ul class="recent-decisions">${recent.map(r=>`<li><i class="decision-dot ${r.review.verdict==='confirmed'?'confirmed':''}"></i><div><strong>${esc(caseTitle(r.sample))}</strong><small>${esc(r.sourceId)} · ${esc(reviewLabel(r.review))}</small></div></li>`).join('')}</ul>`:'<p class="empty-note">Решения ещё не сохранены. Выберите эпизод и проверьте наблюдение.</p>'}<a class="export-link" href="/api/dataset.csv">${icon('download',14)}Экспорт решений CSV</a></section></aside>`;
}
function viewer(s,j,{compact=false}={}){
 const sequence=imageSet(s),sourceImages=imageSources(s),sourceImage=sourceImages[Math.min(selectedImageIndex,Math.max(0,sourceImages.length-1))],r=j?.result,episode=selectedEpisode(j),camera=cameraIdFor(j),videos=cameraVideos(j),selectedVideo=cameraVideo(j),sourceUrl=cameraSource(s,j),renderedOnly=Boolean(selectedVideo&&!selectedVideo.sourceUrl&&selectedVideo.url!==s.video),sourceUnavailable=Boolean(s.video&&camera&&camera!==r?.primaryCamera&&!selectedVideo),cameraFrames=framesForCamera(r,camera),still=activeFrame(j),frames=episode?.frameTimes||[],busy=pending||store.jobs.some(processing);
 const allEpisodes=episodesForResult(r),currentIndex=still?frames.findIndex(t=>Math.abs(t-still.time)<.02):0;
 const previewIndices=[...new Set([0,Math.floor((frames.length-1)/2),frames.length-1])].filter(i=>i>=0);
 const displayedSource=sequence?(sourceImage?.url||s.image):s.image;
 const episodeLabel=(item,index)=>`${item.camera?.replace(/^user_camera_/,'Камера ')||'Камера'} · ${sequence?Number.isInteger(item.image_index)?`Стоп-кадр ${item.image_index+1}`:`Эпизод ${index+1}`:`${Number(item.triggerTime??item.start??0).toFixed(1)} с`}`;
 return `<section class="view-panel ${compact?'camera-viewer':''}" aria-label="Видео и детекции"><div class="viewer-toolbar"><span>${icon(s.video?'camera':'eye',15)}${s.video?'Видеозапись':sequence?'Набор стоп-кадров':'Изображение'} <b>${esc(s.eventId)}</b></span><div>${videos.length>1?`<label class="viewer-select"><span>Камера</span><select id="camera-select" aria-label="Выбрать камеру">${videos.map(video=>`<option value="${esc(video.camera)}" ${video.camera===camera?'selected':''}>${esc(video.camera.replace(/^user_camera_/,'Камера '))}</option>`).join('')}</select></label>`:''}${allEpisodes.length>1?`<label class="viewer-select"><span>Эпизод</span><select id="episode-select" aria-label="Выбрать эпизод">${allEpisodes.map((item,index)=>`<option value="${esc(item.eventId)}" ${item.eventId===episode?.eventId?'selected':''}>${esc(episodeLabel(item,index))}</option>`).join('')}</select></label>`:''}<button class="icon-btn ${boxes?'toggled':''}" data-action="boxes" aria-label="Показать или скрыть рамки объектов" aria-pressed="${boxes}">${icon('target',18)}</button><button class="icon-btn expand-view" data-action="expand" aria-label="Развернуть просмотр" aria-expanded="${viewerExpanded}">${icon(viewerExpanded?'close':'expand',18)}</button></div></div><div class="frame-hud"><span>${esc(byId(s.id)?.sourceId||s.eventId)}${camera?' · '+esc(camera.replace(/^user_camera_/,'Камера ')):''}</span><span><b id="hud-mode">${sequence?'Стоп-кадр':still?'Точный кадр':s.video?'Видео':'Фото'}</b> <span id="frame-time">${sequence?`Кадр ${selectedImageIndex+1} из ${sourceImages.length}`:'00:00.00'}</span></span></div><div class="viewport ${editing?'editing':''}" id="viewport"><div class="frame-canvas" id="frame-canvas">${s.video?`<video id="media" data-sample="${esc(s.id)}" data-camera="${esc(camera)}" data-rendered="${renderedOnly||sourceUnavailable}" src="${esc(sourceUrl||(camera===r?.primaryCamera?s.video:''))}" poster="${esc(selectedVideo?.sourceUrl?'':selectedVideo?.poster||(camera===r?.primaryCamera?s.image:''))}" controls playsinline preload="metadata" aria-label="${esc(s.title)}"></video>`:`<img id="media" data-sample="${esc(s.id)}" data-camera="${esc(camera)}" src="${esc(displayedSource)}" alt="${esc(sequence?`Стоп-кадр ${selectedImageIndex+1} из ${sourceImages.length}`:s.title)}">`}${still?.path?`<img id="evidence-media" class="evidence-media" src="${evidence(still.path)}" alt="Точный кадр разбора на ${still.time.toFixed(2)} секунде">`:''}<svg class="overlay" id="overlay" aria-hidden="true"></svg>${editing?'<div class="edit-hint">Отметьте от 3 до 12 точек границы</div>':''}${sourceUnavailable?'<div class="no-boxes-note">Видео этой камеры загружается</div>':!cameraFrames.length?'<div class="no-boxes-note">Рамки этой камеры появятся после обработки кадров</div>':''}</div></div><div class="playback-row">${s.video?`<button class="btn ${still?'primary':''}" data-action="${still?'video':'play'}">${icon('play',15)}${still?'Смотреть видео':'Воспроизвести'}</button><div class="timeline"><div class="timeband">${episode?'<span class="episode-segment"></span>':''}<span id="time-cursor"></span><input id="timeline" type="range" min="0" max="${duration(s)||1}" step="0.01" value="${still?.time||0}" aria-label="Позиция видео в секундах"></div><span>${duration(s).toFixed(1)} с</span></div>`:sequence?`<button class="btn" type="button" data-image-step="-1" ${selectedImageIndex<=0?'disabled':''}>Предыдущий кадр</button><span class="image-mode">Стоп-кадр ${selectedImageIndex+1} из ${sourceImages.length} · порядок кадров, без движения</span><button class="btn" type="button" data-image-step="1" ${selectedImageIndex>=sourceImages.length-1?'disabled':''}>Следующий кадр</button>`:'<span class="image-mode">Исходное изображение</span>'}<button class="btn quiet" data-action="run" ${busy||!s.video||uploadKind==='image_set'?'disabled':''}>${icon('spark',14)}${!s.video?'Только просмотр':busy?(processing(j)?'Разбираем…':pending?'Запускаем…':'Обработка другого ролика'):j?'Повторить разбор':'Разобрать'}</button></div><div class="frame-readout"><div class="frame-counts"><strong id="track-count">Объекты: —</strong><span><b id="people-count">—</b> людей · <b id="object-count">—</b> предметов</span><small id="sampled-time">${sequence?'Выбранный стоп-кадр':'Кадр разбора'}</small></div><div id="frame-objects" class="frame-objects" role="group" aria-label="Объекты в текущем кадре"></div></div>${episode?.frames?.length?`<div class="evidence-strip" aria-label="Ключевые кадры эпизода">${previewIndices.map(i=>`<button class="evidence-thumb ${i===currentIndex?'current':''}" data-seek="${Number(frames[i])}" aria-label="Открыть доказательство ${i+1}"><img src="${evidence(episode.frames[i])}" alt="Ключевой кадр ${i+1}"><span class="thumb-source">${esc(episode.camera||s.eventId)}</span><time>${sequence?`Кадр ${i+1}`:`${Number(frames[i]).toFixed(2)} с`}</time></button>`).join('')}</div><details class="all-keyframes"><summary>Все ${frames.length} кадров разбора</summary><div class="keyframe-rail">${frames.map((t,index)=>`<button class="keyframe-button" data-seek="${Number(t)}" aria-label="Открыть доказательство ${index+1}">${sequence?`Кадр ${index+1}`:`${Number(t).toFixed(2)} с`}</button>`).join('')}</div></details>`:''}${editing?`<div class="zone-toolbar"><span>${polygon.length} точек · область для будущих запусков; прошлые результаты не меняются</span><button class="btn" data-action="undo" ${!polygon.length?'disabled':''}>Убрать точку</button><button class="btn primary" data-action="save-zone" ${polygon.length<3?'disabled':''}>Сохранить</button><button class="btn" data-action="cancel-zone">Отмена</button></div>`:''}</section>`;
}
function decision(s,j,vr){
 const episode=selectedEpisode(j),draftKey=reviewDraftKey(j?.id,episode?.eventId||'');
 if(!s.video&&!imageSet(s))return '<section class="reason-panel empty-state"><h2>Только просмотр</h2><p>Для анализа загрузите видео.</p></section>';
 if(!j)return `<section class="reason-panel empty-state"><h2>Что происходит в кадре?</h2><p>Запустите разбор, чтобы выделить объекты и описать наблюдаемое действие.</p><button class="btn primary" data-action="run">${icon('play',16)}Запустить разбор</button></section>`;
 if(j.status==='failed'||j.analysisError)return `<section class="reason-panel empty-state"><h2>Разбор не завершён</h2><p>Сохранённые кадры доступны для проверки. Повторите разбор или выберите другое видео.</p><button class="btn" data-action="run">Повторить</button></section>`;
 if(processing(j)||vlmPending(j))return `<section class="reason-panel processing"><span class="spinner"></span><h2>Анализируем событие</h2><p>${j.result?.frames?.length?'Видео с рамками доступно.':'Подготавливаем видео.'}</p></section>`;
 const a=j.analysis,r=j.result;if(!a)return `<section class="reason-panel"><h2>Кадры готовы к проверке</h2><p>Просмотрите источник и найденные объекты.</p></section>`;
 if(isPresenceTask(j.task)){const status=a.assessment==='confirmed'?'Объект обнаружен':a.assessment==='not_confirmed'?'Объект не обнаружен':'Нужна проверка';return `<section class="reason-panel"><div class="reason-heading"><h2>${status}</h2></div><p class="observed-action">Профиль «${esc(taskTitle(j.task))}» проверяет наличие целевого объекта. Действие по этим данным не оценивается.</p></section><section class="operator-review" aria-label="Решение оператора"><label for="review-note">Комментарий к решению <span>необязательно</span></label><textarea id="review-note" data-job="${esc(j.id)}" data-event-id="${esc(episode?.eventId||'')}" maxlength="600" placeholder="Что важно учесть при проверке?">${esc(reviewDrafts.get(draftKey)??vr?.note??'')}</textarea>${vr?`<div class="saved-review">${icon('check',17)}${esc(reviewLabel(vr))}</div>`:''}<div class="review-actions"><button class="btn primary accept" data-verdict="confirmed" ${pending?'disabled':''}>${icon('check',20)}Подтвердить событие</button><button class="btn pass" data-verdict="rejected" ${pending?'disabled':''}>${icon('chevron',20)}Ложный сигнал</button></div><button class="text-button" data-verdict="uncertain" ${pending?'disabled':''}>Недостаточно данных</button><p class="muted-note">Решение сохраняется отдельно от результата модели.</p></section>`;}
 const candidateEvidence=actionEvidencePresentation(a);if(candidateEvidence){const observation=candidateEvidence.observation||{};return `<section class="reason-panel"><div class="reason-heading"><h2>${candidateEvidence.status}</h2></div><p class="observed-action">${esc(candidateEvidence.primary)}</p>${observation.reason||observation.evidence?.length?`<details class="analysis-explanation"><summary>Неподтверждённое наблюдение модели</summary>${observation.reason?`<p>${esc(observation.reason)}</p>`:''}${observation.evidence?.length?`<ul class="evidence-facts">${observation.evidence.map(fact=>`<li>${icon('eye',17)}<span>${esc(fact)}</span></li>`).join('')}</ul>`:''}</details>`:''}</section><section class="operator-review" aria-label="Решение оператора"><label for="review-note">Комментарий к решению <span>необязательно</span></label><textarea id="review-note" data-job="${esc(j.id)}" data-event-id="${esc(episode?.eventId||'')}" maxlength="600" placeholder="Что важно учесть при проверке?">${esc(reviewDrafts.get(draftKey)??vr?.note??'')}</textarea>${vr?`<div class="saved-review">${icon('check',17)}${esc(reviewLabel(vr))}</div>`:''}<div class="review-actions"><button class="btn primary accept" data-verdict="confirmed" ${pending?'disabled':''}>${icon('check',20)}Подтвердить событие</button><button class="btn pass" data-verdict="rejected" ${pending?'disabled':''}>${icon('chevron',20)}Ложный сигнал</button></div><button class="text-button" data-verdict="uncertain" ${pending?'disabled':''}>Недостаточно данных</button><p class="muted-note">Решение сохраняется отдельно от результата модели.</p></section>`;}
 const confirmed=a.assessment==='confirmed',title=confirmed?'Почему сработало':a.assessment==='not_confirmed'?'Почему не подтверждено':'Что требует проверки';
 const steps=[['Источник',s.video?duration(s).toFixed(1)+' с':imageSet(s)?`${imageSources(s).length} стоп-кадров`:'Один кадр'],['Кадры',`${r?.frames?.length||0} для разбора`],['Разбор','Результат готов'],['Оператор',vr?'Решение сохранено':'Ждёт решения']];
 return `<section class="reason-panel"><div class="reason-heading"><h2>${title}</h2></div><p class="observed-action">${esc(a.action||a.summary||'Проверьте событие')}</p>${a.facts?.length?`<ul class="evidence-facts">${a.facts.map(f=>`<li>${icon('check',17)}<span>${esc(f)}</span></li>`).join('')}</ul>`:''}<details class="analysis-explanation"><summary>Полное объяснение</summary><p>${esc(a.summary)}</p></details></section><section class="decision-route"><h2>Маршрут решения</h2><ol>${steps.map(([title,sub],i)=>`<li class="${i===3?'last-step':''}"><span>${i+1}</span><strong>${title}</strong><small>${esc(sub)}</small></li>`).join('')}</ol></section><section class="operator-review" aria-label="Решение оператора"><label for="review-note">Комментарий к решению <span>необязательно</span></label><textarea id="review-note" data-job="${esc(j.id)}" data-event-id="${esc(episode?.eventId||'')}" maxlength="600" placeholder="Что важно учесть при проверке?">${esc(reviewDrafts.get(draftKey)??vr?.note??'')}</textarea>${vr?`<div class="saved-review">${icon('check',17)}${esc(reviewLabel(vr))}</div>`:''}<div class="review-actions"><button class="btn primary accept" data-verdict="confirmed" ${pending?'disabled':''}>${icon('check',20)}Подтвердить событие</button><button class="btn pass" data-verdict="rejected" ${pending?'disabled':''}>${icon('chevron',20)}Ложный сигнал</button></div><button class="text-button" data-verdict="uncertain" ${pending?'disabled':''}>Недостаточно данных</button><p class="muted-note">Решение сохраняется отдельно от результата анализа.</p></section><div class="source-attribution">${icon('link',14)}<a href="${safeLink(s.sourceUrl)}" target="_blank" rel="noreferrer">${esc(s.source||'Источник видео')}</a><span>${esc(s.license||'')}</span></div>`;
}
function uploadControls(busy){
 const ready=service.vlm_ready??service.configured??false;
 const enabled=uploadUseVlm===null?ready:uploadUseVlm&&ready;
 const source=sample(),canRun=uploadKind==='image_set'?uploadFiles.length>0:uploadFiles.length>0||Boolean(source?.video);
 const compatible=uploadKind==='image_set'?uploadTasks.filter(task=>['phone_objects','cigarette_objects'].includes(task.id)):uploadTasks;
 const selectedTask=taskProfile(uploadTask);
 const limit=uploadKind==='video'?uploadFiles.length<=2:uploadFiles.length>=2&&uploadFiles.length<=24;
 const fileLabel=uploadFiles.length?uploadFiles.map(file=>esc(file.name)).join(' · '):uploadKind==='video'?(source?.video?esc(source.title||'Выбранное видео'):'Одно или два видео'):'Выберите от 2 до 24 стоп-кадров';
 return `<form id="upload-form" class="zone-toolbar upload-toolbar" aria-label="Загрузить материалы"><label class="btn" for="upload-videos">${icon('upload',17)}Выбрать видео<input id="upload-videos" type="file" accept="video/*,.mp4,.mov,.mkv,.avi,.webm" multiple class="sr-only" ${pending?'disabled':''}></label><label class="btn" for="upload-images">Выбрать стоп-кадры<input id="upload-images" type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" multiple class="sr-only" ${pending?'disabled':''}></label><span class="upload-filenames">${fileLabel}</span><label class="upload-task-label">Профиль обработки<select id="upload-task" ${pending?'disabled':''}>${compatible.map(task=>`<option value="${esc(task.id)}" ${task.id===uploadTask?'selected':''}>${esc(task.title_ru||task.id)}</option>`).join('')}</select>${selectedTask?`<small>${isPresenceTask(selectedTask.id)?'Наличие предмета':'Действие по видео · кандидат для проверки'}</small>`:''}${uploadKind==='image_set'?'<small>Объекты по отдельным изображениям; порядок кадров не означает движение.</small>':''}</label>${uploadKind==='video'?`<label class="upload-analysis"><input id="upload-vlm" type="checkbox" ${enabled?'checked':''} ${!ready||pending?'disabled':''}>Анализ событий${!ready?'<span class="sr-only">Временно недоступен</span>':''}</label>`:''}<button class="btn primary" type="submit" ${busy||pending||!canRun||!limit||!compatible.length||service.linq_available===false?'disabled':''}>${pending?'Загрузка…':'Запустить'}</button>${uploadError?`<span class="upload-error" role="alert">${esc(uploadError)}</span>`:''}</form>`;
}
function monitor(){
 const s=sample();if(!s)return uploadControls(store.jobs.some(processing))+'<div class="card empty-state"><h1>Нет источников</h1><p>Загрузите видео для проверки.</p></div>';
 const j=latest(s),r=byId(s.id),episode=selectedEpisode(j),vr=reviewForSelection(store.reviews,j?.id,episode?.eventId||'');
 return uploadControls(store.jobs.some(processing))+`<div class="incident-layout">${queue()}<section class="incident-detail" aria-label="Карточка инцидента"><article class="card incident-card"><div class="incident-heading"><div><h1><i style="background:${CASE_COLORS[s.case]||'#487bcc'}"></i>${esc(caseTitle(s))}</h1><p>${esc(s.eventId)} · ${esc(r?.sourceId||'')} · ${esc(s.title)}${r?.timestamp?' · '+stamp(r.timestamp):''}</p></div><div class="incident-badges"><span class="status-pill ${j?.analysis?.assessment==='unknown'?'attention':''}">${icon(j?.analysis?.assessment==='confirmed'?'warning':'eye',13)}${esc(s.video?assessmentLabel(j):'Только просмотр')}</span><span class="status-pill">${esc(reviewLabel(vr))}</span></div></div>${viewer(s,j)}${decision(s,j,vr)}</article></section>${summary()}</div>`;
}
function analytics(){return renderAnalytics({records:records(),periodDays,now:Date.now(),matrixMode});}
function cameras(){return renderCameras({records:records(),selectedId:selected,viewerMarkup:sample()?viewer(sample(),latest(sample()),{compact:true}):'',caseFilter:cameraFilter,sort:cameraSort});}

function render(){
 const viewerFocus=viewerExpanded?snapshotViewerFocus():null;
 cancelAnimationFrame(animationId);
 const note=$('#review-note');if(note?.dataset.job)reviewDrafts.set(reviewDraftKey(note.dataset.job,note.dataset.eventId),note.value);
 const sourceScroll=$('.sources')?.scrollTop||0,sourceScrollX=$('.sources')?.scrollLeft||0;
 shell(({monitor,events:eventJournal,analytics,cameras}[route()]||monitor)());
 const picker=$('#upload-videos');if(picker&&!uploadFiles.length)picker.value='';
 const next=$('#media');
 if(next&&!boundMedia.has(next)){
  boundMedia.add(next);
  if(next.tagName==='VIDEO'){
   next.addEventListener('loadedmetadata',paint);
   next.addEventListener('timeupdate',paint);next.addEventListener('seeked',paint);
   next.addEventListener('play',()=>{updatePlayback();animate();});next.addEventListener('pause',()=>{updatePlayback();paint();});next.addEventListener('ended',updatePlayback);
  }else next.addEventListener('load',paint);
 }
 $('#evidence-media')?.addEventListener('load',paint,{once:true});
 if($('.sources')){$('.sources').scrollTop=sourceScroll;$('.sources').scrollLeft=sourceScrollX;if(window.innerWidth<=600&&!sourceScrollX){const active=$('.source.selected');if(active) $('.sources').scrollLeft=Math.max(0,active.offsetLeft-$('.sources').offsetLeft);}}
 syncViewerMode();paint();restoreViewerFocus(viewerFocus);
}
function setBoxes(value){boxes=value;const b=$('[data-action="boxes"]');b?.classList.toggle('toggled',boxes);b?.setAttribute('aria-pressed',String(boxes));if(b?.querySelector('span'))b.querySelector('span').textContent=`Детекции ${boxes?'вкл.':'выкл.'}`;paint();}
function snapshotViewerFocus(){
 const panel=$('.view-panel'),el=document.activeElement;if(!panel?.contains(el))return null;
 return {id:el.id,objectKey:el.dataset.objectKey,action:el.dataset.action,seek:el.dataset.seek,keyframe:el.classList.contains('keyframe-button')};
}
function restoreViewerFocus(state){
 if(!viewerExpanded)return;const panel=$('.view-panel');if(!panel)return;
 let target=state?.id?panel.querySelector(`#${CSS.escape(state.id)}`):state?.objectKey?panel.querySelector(`[data-object-key="${CSS.escape(state.objectKey)}"]`):state?.seek?panel.querySelector(`${state.keyframe?'.keyframe-button':''}[data-seek="${CSS.escape(state.seek)}"]`):state?.action?panel.querySelector(`[data-action="${CSS.escape(state.action)}"]`):null;
 const visible=el=>el&&el.getClientRects().length&&getComputedStyle(el).visibility!=='hidden'&&!el.disabled;
 if(!visible(target)){if(panel.contains(document.activeElement)&&visible(document.activeElement))return;target=[...panel.querySelectorAll('button,input,a,summary,textarea,video')].find(visible);}
 target?.focus({preventScroll:true});
}
function syncViewerMode(){
 const panel=$('.view-panel');if(!panel)viewerExpanded=false;
 document.body.classList.toggle('viewer-is-expanded',viewerExpanded);
 document.querySelectorAll('.console-header,.source-panel,.session-summary,.console-footer,.incident-heading,.reason-panel,.decision-route,.operator-review,.source-attribution,.camera-page-heading,.camera-summary,.camera-controls-row,.camera-selected-info,.camera-catalog').forEach(el=>el.inert=viewerExpanded);
 if(panel){panel.classList.toggle('viewer-expanded',viewerExpanded);panel.setAttribute('role',viewerExpanded?'dialog':'region');if(viewerExpanded)panel.setAttribute('aria-modal','true');else panel.removeAttribute('aria-modal');}
 const backdrop=$('.viewer-backdrop');if(backdrop)backdrop.hidden=!viewerExpanded;
 const button=$('[data-action="expand"].expand-view');if(button){button.setAttribute('aria-expanded',String(viewerExpanded));button.setAttribute('aria-label',viewerExpanded?'Свернуть просмотр':'Развернуть просмотр');button.innerHTML=icon(viewerExpanded?'close':'expand',18);}
}
function setViewerExpanded(value){viewerExpanded=value;syncViewerMode();requestAnimationFrame(()=>{paint();$('.expand-view')?.focus({preventScroll:true});});}
document.addEventListener('keydown',e=>{
 if(!viewerExpanded)return;
 if(e.key==='Escape'){e.preventDefault();setViewerExpanded(false);return;}
 if(e.key==='Tab'){const controls=[...$('.view-panel').querySelectorAll('button,a,input,summary,textarea')].filter(el=>!el.disabled&&el.getClientRects().length);if(!controls.length)return;const first=controls[0],last=controls.at(-1);if(!$('.view-panel').contains(document.activeElement)){e.preventDefault();(e.shiftKey?last:first).focus();return;}if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}
});
function updatePlayback(){const m=$('#media'),b=$('[data-action="play"]');if(!b||!m)return;const paused=Boolean(activeFrame(latest(sample())))||m.paused;b.innerHTML=icon(paused?'play':'pause',15);b.setAttribute('aria-label',paused?'Воспроизвести видео':'Приостановить видео');}
function animate(){cancelAnimationFrame(animationId);paint();if($('#media')?.tagName==='VIDEO'&&!$('#media').paused)animationId=requestAnimationFrame(animate);}
export function renderFrameAt(frames,time,{staticImage=false,exact=false,maxAge=1.5}={}){
 const ordered=(frames||[]).filter(f=>Number.isFinite(f.time)).slice().sort((a,b)=>a.time-b.time);
 if(!ordered.length)return {frame:null,interpolated:false};
 if(staticImage)return {frame:ordered[0],interpolated:false};
 if(exact)return {frame:ordered.find(f=>Math.abs(f.time-time)<.02)||null,interpolated:false};
 const frame=ordered.reduce((best,f)=>f.time<=time+.02?f:best,null);
 return {frame:frame&&time<=frame.time+maxAge?frame:null,interpolated:false};
}
const edges=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
function paint(){
 const svg=$('#overlay'),viewport=$('#frame-canvas'),media=$('#media');if(!svg||!media)return;
 const image=$('#evidence-media'),display=image||media;
 const W=viewport.clientWidth,H=viewport.clientHeight,mw=display.videoWidth||display.naturalWidth||16,mh=display.videoHeight||display.naturalHeight||9;
 const scale=Math.min(W/mw,H/mh),w=mw*scale,h=mh*scale,x=(W-w)/2,y=(H-h)/2;
 const currentSample=sample(),j=latest(currentSample),r=j?.result,episode=selectedEpisode(j),camera=cameraIdFor(j),sequence=imageSet(currentSample),frames=sequence?framesForImageIndex(r,camera,selectedImageIndex):framesForCamera(r,camera),still=activeFrame(j),time=still?.time??(media.currentTime||0);
 const {frame,interpolated}=renderFrameAt(frames,time,{staticImage:!sample().video,exact:Boolean(still),maxAge:1.5/(r?.sampleFrequency||1)});
 const zone=editing?polygon:(store.zones[selected]||r?.zone);
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 let contents=zone?.length?`<polygon points="${zone.map(([a,b])=>`${x+a*w},${y+b*h}`).join(' ')}" class="zone-shape"/>`:'';
 if(editing)contents+=polygon.map(([a,b])=>`<circle cx="${x+a*w}" cy="${y+b*h}" r="5" fill="#80b4ff" stroke="#18243a"/>`).join('');
 const annotations=placeAnnotations(frame?.detections||[],{x,y,w,h},sample().case,focusedObject);
 if(boxes&&!editing&&frame&&!(media.dataset.rendered==='true'&&!image)){
  contents+=annotations.slice().reverse().map(a=>{
   const d=a.d,pose=d.pose?edges.map(([u,v])=>d.pose[u]&&d.pose[v]&&(d.poseConfidence?.[u]??0)>.3&&(d.poseConfidence?.[v]??0)>.3?`<line x1="${x+d.pose[u][0]*w}" y1="${y+d.pose[u][1]*h}" x2="${x+d.pose[v][0]*w}" y2="${y+d.pose[v][1]*h}" stroke="${a.color}" stroke-width="1.5"/>`:'').join(''):'';
   return `<g><title>${esc(a.text)}</title><rect class="detection-box ${a.selected?'selected':''} ${a.related?'related':''}" data-object-box="${esc(a.key)}" x="${a.x}" y="${a.y}" width="${a.w}" height="${a.h}" fill="${a.selected?'#ffffff0c':'none'}" stroke="${a.color}" stroke-width="${a.selected?3.2:a.related?2.4:1.7}"/>${pose}${a.related||a.selected?`<path d="M${a.x} ${a.y+8}v-8h8M${a.x+a.w-8} ${a.y}h8v8M${a.x} ${a.y+a.h-8}v8h8M${a.x+a.w-8} ${a.y+a.h}h8v-8" fill="none" stroke="${a.color}" stroke-width="2.8"/>`:''}</g>`;
  }).join('');
  contents+=annotations.map(a=>`<g class="object-annotation"><path d="M${a.leader.from.join(' ')}L${a.leader.to.join(' ')}" fill="none" stroke="${a.color}" stroke-width="1.1" opacity=".85"/><rect class="annotation-label" x="${a.tag.x}" y="${a.tag.y}" width="${a.tag.w}" height="${a.tag.h}" rx="3" fill="#15171df2" stroke="${a.color}" stroke-width="${a.selected?1.6:1}"/><circle cx="${a.tag.x+9}" cy="${a.tag.y+12.5}" r="2.4" fill="${a.color}"/><text x="${a.tag.x+17}" y="${a.tag.y+17}" fill="#ffffff" font-size="${a.font}" font-family="system-ui" font-weight="${a.related||a.selected?650:550}">${esc(a.text)}</text></g>`).join('');
 }
 svg.innerHTML=contents;
 $('#frame-time').textContent=sequence?`Кадр ${selectedImageIndex+1} из ${imageSources(currentSample).length}`:clock(time);
 $('#track-count').textContent=frame?'Объекты: '+annotations.length:'Объекты: —';
 $('#hud-mode').textContent=sequence?'Стоп-кадр':still?'Точный кадр':!sample().video?'Фото':interpolated?'Интерполяция':'Видео';
 const objects=$('#frame-objects'),objectMarkup=annotations.map(a=>`<button data-object-key="${esc(a.key)}" class="frame-object ${a.related?'related':''} ${a.selected?'selected':''}" aria-pressed="${a.selected}" aria-label="Выделить ${esc(a.text)}" style="--object-color:${a.color}"><i></i>${esc(a.text)}</button>`).join('')||`<span class="objects-empty">${r?.frames?.length?'На этом моменте нет доступных рамок':'Объекты появятся после обработки кадров'}</span>`;
 if(objects.dataset.markup!==objectMarkup){const focus=objects.contains(document.activeElement)?snapshotViewerFocus():null;objects.innerHTML=objectMarkup;objects.dataset.markup=objectMarkup;if(focus)restoreViewerFocus(focus);}
 document.querySelectorAll('.keyframe-button').forEach(b=>{const current=Boolean(still)&&Math.abs(Number(b.dataset.seek)-time)<.02;b.classList.toggle('current',current);if(current)b.setAttribute('aria-current','true');else b.removeAttribute('aria-current');});
 $('#people-count').textContent=frame?annotations.filter(a=>a.d.label==='person').length:'—';
 $('#object-count').textContent=frame?annotations.filter(a=>a.d.label!=='person').length:'—';
 $('#sampled-time').textContent=sequence?frame?`Стоп-кадр ${selectedImageIndex+1}`:'Для этого стоп-кадра рамки недоступны':frames.length?(interpolated?'Между кадрами — интерполяция рамок':frame?`Кадр ${camera?camera.replace(/^user_camera_/,'камеры '):''}: ${frame.time.toFixed(2)} с`:'Между кадрами: рамки скрыты'):r?'Для этой камеры кадры разбора недоступны':'Ожидает запуска';
 if($('#time-cursor'))$('#time-cursor').style.left=Math.min(100,100*time/(duration(sample())||1))+'%';
 if($('#timeline')&&document.activeElement!==$('#timeline'))$('#timeline').value=time;
 updatePlayback();
}
async function refresh(){
 const [next,status,catalog]=await Promise.all([api('/api/pipeline'),api('/api/status'),api('/api/samples')]);
 const previousEvents=JSON.stringify(events),previousOptions=JSON.stringify(eventOptions);
 if(route()==='events'){
  try{const [all,response]=await Promise.all([api('/api/events'),api(eventRequestUrl())]);eventOptions=Array.isArray(all.events)?all.events:[];events=Array.isArray(response.events)?response.events:[];eventError='';}
  catch(error){eventError=error.message||'Не удалось загрузить журнал событий.';}
 }
 if(JSON.stringify(next)!==JSON.stringify(store)||JSON.stringify(status)!==JSON.stringify(service)||JSON.stringify(catalog.samples)!==JSON.stringify(samples)||previousEvents!==JSON.stringify(events)||previousOptions!==JSON.stringify(eventOptions)){store=next;service=status;samples=catalog.samples;render();}
}
function eventRequestUrl(){const query=new URLSearchParams();if(eventFilter.camera)query.set('camera',eventFilter.camera);if(eventFilter.type)query.set('type',eventFilter.type);if(eventFilter.zone)query.set('zone',eventFilter.zone);if(eventFilter.start!=='')query.set('start_s',eventFilter.start);if(eventFilter.end!=='')query.set('end_s',eventFilter.end);return '/api/events'+(query.size?'?'+query.toString():'');}
async function act(action){
 if(action==='run'){
  if(!sample()?.video||uploadKind==='image_set')return;
  if(pending||store.jobs.some(processing))return;
  pending=true;selectedJobId='';render();
  try{await api('/api/pipeline',{sampleId:selected,task:uploadTask,use_vlm:Boolean((uploadUseVlm===null?service.vlm_ready??service.configured:uploadUseVlm)&&Boolean(service.vlm_ready??service.configured))});await refresh();}catch(e){toast(e.message||'Не удалось выполнить действие. Попробуйте ещё раз.');}finally{pending=false;render();}
 }else if(action==='expand'){setViewerExpanded(!viewerExpanded);}
 else if(action==='boxes'){setBoxes(!boxes);}
 else if(action==='play'||action==='video'){const m=$('#media');if(m?.tagName==='VIDEO'){if(activeFrame(latest(sample()))){const time=frameView.time;frameView=null;render();const next=$('#media'),play=()=>{next.currentTime=time;next.play().catch(()=>toast('Не удалось воспроизвести видео. Используйте элементы управления плеером.'));};if(next.readyState>=1)play();else next.addEventListener('loadedmetadata',play,{once:true});}else if(m.paused)m.play().catch(()=>toast('Не удалось воспроизвести видео. Используйте элементы управления плеером.'));else m.pause();}}
 else if(action==='zone'){editing=true;polygon=[...(store.zones[selected]||[])];$('#media')?.pause?.();render();}
 else if(action==='undo'){polygon.pop();render();}
 else if(action==='cancel-zone'){editing=false;render();}
 else if(action==='save-zone'){
  try{const response=await api('/api/zones',{sampleId:selected,polygon});editing=false;await refresh();render();toast(response.inferenceEnabledForFutureJobs?'Область сохранена для будущих запусков. Текущие результаты не меняются.':'Область сохранена. Она не меняет существующие результаты.');}catch(e){toast(e.message||'Не удалось выполнить действие. Попробуйте ещё раз.');}
 }
}
function selectSource(id,jobId='',cameraId='',eventId='',imageIndex=0,triggerTime=null){if(!samples.some(s=>s.id===id))return;const chosen=samples.find(s=>s.id===id),task=taskForCase(chosen.case);if(uploadTasks.some(item=>item.id===task))uploadTask=task;selectedJobId=jobId;selectedCameraId=cameraId;selectedEventId=eventId;selectedImageIndex=Math.max(0,Number(imageIndex)||0);focusedObject='';selected=id;kind=sample().video?'video':'image';editing=false;const hasTriggerTime=typeof triggerTime==='number'&&Number.isFinite(triggerTime);frameView=sample().video?sampledFrame(latest(sample()),hasTriggerTime?triggerTime:0,hasTriggerTime):null;location.hash='monitor';render();if(sample().video&&hasTriggerTime){const media=$('#media'),time=frameView?.time??triggerTime,seek=()=>{media.pause();media.currentTime=time;};if(media?.tagName==='VIDEO'){if(media.readyState>=1)seek();else media.addEventListener('loadedmetadata',seek,{once:true});}}}
document.addEventListener('click',async e=>{
 const queueButton=e.target.closest('[data-queue-filter]');if(queueButton){queueFilter=queueButton.dataset.queueFilter;render();return;}
 const period=e.target.closest('[data-period]');if(period){periodDays=Number(period.dataset.period);render();return;}
 const matrix=e.target.closest('[data-matrix]');if(matrix){matrixMode=matrix.dataset.matrix;render();return;}
 const cameraCase=e.target.closest('[data-camera-case]');if(cameraCase){cameraFilter=cameraCase.dataset.cameraCase;render();return;}
 const cameraOrder=e.target.closest('[data-camera-sort]');if(cameraOrder){cameraSort=cameraOrder.dataset.cameraSort;render();return;}
 const cameraSource=e.target.closest('[data-camera-source]');if(cameraSource){selected=cameraSource.dataset.cameraSource;selectedJobId='';selectedCameraId='';selectedEventId='';focusedObject='';editing=false;frameView=sample().video?sampledFrame(latest(sample())):null;render();const panel=$('.camera-selected');panel?.setAttribute('tabindex','-1');panel?.focus({preventScroll:true});panel?.scrollIntoView({block:'start',behavior:'instant'});return;}
 const imageStep=e.target.closest('[data-image-step]');if(imageStep){selectedImageIndex=Math.max(0,Math.min(imageSources(sample()).length-1,selectedImageIndex+Number(imageStep.dataset.imageStep)));focusedObject='';render();return;}
 const eventOpen=e.target.closest('[data-event-open]');if(eventOpen){const item=events.find(value=>value.event_id===eventOpen.dataset.eventOpen&&value.job_id===eventOpen.dataset.job);if(item){const job=store.jobs.find(value=>value.id===item.job_id);selectSource(item.sample_id||job?.sampleId,item.job_id,item.camera_id,item.event_id,imageIndexForEvidence(item),Number.isFinite(item.trigger_timestamp_s)?item.trigger_timestamp_s:null);}return;}
 const eventReviewButton=e.target.closest('[data-event-review]');if(eventReviewButton){if(pending)return;const id=eventReviewButton.dataset.eventId,jobId=eventReviewButton.dataset.job,note=$(`[data-event-note="${CSS.escape(id)}"]`)?.value||'';pending=true;render();try{await fetch(`/api/events/${encodeURIComponent(jobId)}/${encodeURIComponent(id)}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmed:eventReviewButton.dataset.eventReview==='true',note})}).then(async response=>{const value=await response.json();if(!response.ok)throw new Error(value.detail||value.error?.message||'Не удалось сохранить решение');return value;});eventReviewDrafts.delete(id);await refresh();toast('Решение по событию сохранено');}catch(error){eventError=error.message;toast(error.message);}finally{pending=false;render();}return;}
 const source=e.target.closest('[data-source]');if(source){if(source.dataset.job)caseFilter='all';selectSource(source.dataset.source,source.dataset.job);return;}
 const type=e.target.closest('[data-kind]');if(type){const first=samples.find(s=>Boolean(s.video)===(type.dataset.kind==='video'));if(first){caseFilter='all';selectSource(first.id);}return;}
 const filter=e.target.closest('[data-case]');if(filter){caseFilter=filter.dataset.case;const first=samples.find(s=>Boolean(s.video)===(kind==='video')&&(caseFilter==='all'||s.case===caseFilter));if(first&&!(caseFilter==='all'||sample().case===caseFilter))selectSource(first.id);else render();return;}
 const object=e.target.closest('[data-object-key]');if(object){focusedObject=focusedObject===object.dataset.objectKey?'':object.dataset.objectKey;setBoxes(true);$('#frame-objects')?.querySelector(`[data-object-key="${CSS.escape(object.dataset.objectKey)}"]`)?.focus({preventScroll:true});return;}
 const action=e.target.closest('[data-action]');if(action){act(action.dataset.action);return;}
 const seek=e.target.closest('[data-seek]');if(seek){const m=$('#media');if(m?.tagName==='VIDEO'){frameView=sampledFrame(latest(sample()),Number(seek.dataset.seek));m.pause();render();}return;}
 const verdict=e.target.closest('[data-verdict]');if(verdict){if(pending)return;pending=true;const j=latest(sample()),episode=selectedEpisode(j),eventId=episode?.eventId||'',draftKey=reviewDraftKey(j.id,eventId),note=$('#review-note')?.value||'';try{await api('/api/reviews',reviewPayload({jobId:j.id,eventId,verdict:verdict.dataset.verdict,note}));reviewDrafts.delete(draftKey);await refresh();toast('Решение сохранено');}catch(e){toast(e.message||'Не удалось выполнить действие. Попробуйте ещё раз.');}finally{pending=false;render();}return;}
 if(editing&&e.target.closest('#frame-canvas')){
  const m=$('#evidence-media')||$('#media'),v=$('#frame-canvas'),b=v.getBoundingClientRect(),W=v.clientWidth,H=v.clientHeight,mw=m.videoWidth||m.naturalWidth||16,mh=m.videoHeight||m.naturalHeight||9;
  const scale=Math.min(W/mw,H/mh),w=mw*scale,h=mh*scale;
  const x=(e.clientX-b.left-(W-w)/2)/w,y=(e.clientY-b.top-(H-h)/2)/h;
  if(x>=0&&x<=1&&y>=0&&y<=1&&polygon.length<12){polygon.push([+x.toFixed(4),+y.toFixed(4)]);render();}
 }
});
document.addEventListener('change',event=>{
 if(event.target.id==='upload-videos'){
  const files=[...event.target.files];
  uploadKind='video';uploadFiles=files;uploadError=files.length>2?'Выберите не больше двух видео.':'';
  const compatible=uploadTasks;
  if(!compatible.some(task=>task.id===uploadTask))uploadTask=compatible[0]?.id||'';
  render();
 }else if(event.target.id==='upload-images'){
  const files=[...event.target.files];
  uploadKind='image_set';uploadFiles=files;
  uploadError=files.length<2||files.length>24?'Выберите от 2 до 24 изображений.':files.some(file=>!isStillImage(file))?'Поддерживаются только JPEG и PNG.':'';
  const compatible=uploadTasks.filter(task=>['phone_objects','cigarette_objects'].includes(task.id));
  if(!compatible.some(task=>task.id===uploadTask))uploadTask=compatible[0]?.id||'';
  render();
 }else if(event.target.id==='upload-task')uploadTask=event.target.value;
 else if(event.target.id==='upload-vlm')uploadUseVlm=event.target.checked;
});
document.addEventListener('submit',async event=>{
 if(event.target.id!=='upload-form')return;
 event.preventDefault();if(pending||store.jobs.some(processing)||service.linq_available===false)return;
 if(!uploadFiles.length){await act('run');return;}
 if(uploadKind==='video'&&uploadFiles.length>2){uploadError='Выберите не больше двух видео.';render();return;}
 if(uploadKind==='image_set'&&(uploadFiles.length<2||uploadFiles.length>24||uploadFiles.some(file=>!isStillImage(file)))){uploadError='Выберите от 2 до 24 изображений JPEG или PNG.';render();return;}
 if(uploadKind==='image_set'&&!['phone_objects','cigarette_objects'].includes(uploadTask)){uploadError='Для набора стоп-кадров выберите профиль наличия предмета.';render();return;}
 const previous=new Set(samples.map(source=>source.id));
 const body=new FormData();uploadFiles.forEach(file=>body.append('files',file));body.append('task',uploadTask);if(uploadKind==='video'){body.append('V','1');body.append('Vfreq','1');}body.append('use_vlm',String(uploadKind==='video'&&(uploadUseVlm===null?service.vlm_ready??service.configured:uploadUseVlm)&&Boolean(service.vlm_ready??service.configured)));
 pending=true;uploadError='';render();
 try{
  const response=await fetch(uploadKind==='image_set'?'/api/image-jobs':'/api/jobs',{method:'POST',body});const receipt=await response.json();
  if(!response.ok)throw new Error(receipt.error?.message||(typeof receipt.detail==='string'?receipt.detail:null)||(typeof receipt.error==='string'?receipt.error:null)||'Не удалось загрузить видео. Повторите запрос.');
  [samples,store]=await Promise.all([api('/api/samples').then(result=>result.samples),api('/api/pipeline')]);
  const value=receipt.job||receipt;
  selected=value.sampleId||value.sample_id||store.jobs.find(job=>job.id===value.id||job.job_id===value.id)?.sampleId||samples.find(source=>!previous.has(source.id))?.id||selected;
  selectedJobId=value.id||value.job_id||'';frameView=null;selectedCameraId='';selectedEventId='';selectedImageIndex=0;kind=uploadKind==='video'?'video':'image';editing=false;uploadFiles=[];const videoPicker=$('#upload-videos'),imagePicker=$('#upload-images');if(videoPicker)videoPicker.value='';if(imagePicker)imagePicker.value='';location.hash='monitor';toast(uploadKind==='image_set'?'Набор стоп-кадров принят.':'Видео загружено. Разбор начался.');
 }catch(error){uploadError=error.message;toast(error.message);}finally{pending=false;render();}
});
document.addEventListener('input',e=>{
 if(e.target.id==='event-search')document.querySelectorAll('[data-search]').forEach(row=>row.hidden=!row.dataset.search.includes(e.target.value.toLowerCase()));
 if(e.target.matches('[data-event-note]'))eventReviewDrafts.set(e.target.dataset.eventNote,e.target.value);
 if(e.target.id==='timeline'){const m=$('#media');if(m?.tagName==='VIDEO'){if(m.paused){const target=Number(e.target.value),exact=sampledFrame(latest(sample()),target);if(exact){frameView=exact;render();}else{frameView=null;m.currentTime=target;paint();}}else{frameView=null;m.currentTime=Number(e.target.value);paint();}}}
 if(e.target.id==='review-note'){const j=latest(sample());if(j)reviewDrafts.set(reviewDraftKey(j.id,e.target.dataset.eventId),e.target.value);}
});
document.addEventListener('change',e=>{if(e.target.id==='model-select'){selectedModel=e.target.value;selectedJobId='';frameView=sample().video?sampledFrame(latest(sample())):null;render();}});
document.addEventListener('change',e=>{
 if(e.target.id==='camera-select'){
  selectedCameraId=e.target.value;const j=latest(sample());
  if(!selectedEpisode(j)){const candidate=episodesForResult(j?.result).find(item=>item.camera===selectedCameraId);selectedEventId=candidate?.eventId||'';}
  frameView=sample().video?sampledFrame(j):null;render();
 }else if(e.target.id==='episode-select'){
  const j=latest(sample()),episode=episodesForResult(j?.result).find(item=>item.eventId===e.target.value);
  selectedEventId=e.target.value;selectedCameraId=episode?.camera||selectedCameraId;if(imageSet(sample()))selectedImageIndex=imageIndexForEvidence(episode,selectedImageIndex);frameView=sample().video?sampledFrame(j):null;render();
 }
});
document.addEventListener('submit',async event=>{
 if(event.target.id!=='event-filters')return;
 event.preventDefault();const start=$('#filter-start').value,end=$('#filter-end').value;
 if((start!==''&&(!Number.isFinite(Number(start))||Number(start)<0))||(end!==''&&(!Number.isFinite(Number(end))||Number(end)<0))||(start!==''&&end!==''&&Number(end)<Number(start))){eventError='Укажите корректный интервал времени в секундах.';render();return;}
 eventFilter={camera:$('#filter-camera').value,type:$('#filter-type').value,zone:$('#filter-zone').value,start,end};
 try{const response=await api(eventRequestUrl());events=Array.isArray(response.events)?response.events:[];eventError='';}catch(error){eventError=error.message||'Не удалось загрузить журнал событий.';}render();
});
window.addEventListener('hashchange',()=>{editing=false;setViewerExpanded(false);render();});
window.addEventListener('resize',paint);
try{
 [samples,service,store,uploadTasks,eventOptions]=await Promise.all([api('/api/samples').then(r=>r.samples),api('/api/status'),api('/api/pipeline'),api('/api/tasks').then(r=>r.tasks).catch(()=>[]),api('/api/events').then(r=>r.events).catch(()=>[])]);events=eventOptions;if(!uploadTasks.some(task=>task.id===uploadTask))uploadTask=uploadTasks[0]?.id||'';
 selected=samples.find(s=>s.id==='proximity-01'&&s.video&&store.jobs.some(j=>j.sampleId===s.id&&j.result?.frames?.length))?.id||samples.find(s=>s.video&&store.jobs.some(j=>j.sampleId===s.id&&j.result?.frames?.length))?.id||samples.find(s=>s.id==='proximity-01'&&s.video)?.id||samples.find(s=>s.video)?.id||samples[0]?.id;kind=sample()?.video?'video':'image';selectedModel=service.model||'';
 const initialTask=taskForCase(sample()?.case);if(uploadTasks.some(task=>task.id===initialTask))uploadTask=initialTask;
 frameView=sample()?.video?sampledFrame(latest(sample())):null;render();setInterval(()=>{if(refreshing)return;refreshing=true;refresh().catch(error=>console.error('LinQ workspace refresh failed',error)).finally(()=>{refreshing=false;});},1000);
}catch(error){console.error('LinQ workspace initialization failed',error);$('#app').innerHTML='<main class="connection-error"><h1>Не удалось загрузить рабочее место</h1><p>Обновите страницу. Если ошибка повторится, проверьте связь с сервером.</p><button class="btn primary" onclick="location.reload()">Обновить страницу</button></main>';}
