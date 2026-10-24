import { CASE_NAMES, CASE_COLORS, summarize } from './console-data.js';

const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const fmt = value => new Intl.NumberFormat('ru-RU').format(value);
const pct = value => value == null ? '—' : `${new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 }).format(value)}%`;
const labelCase = key => CASE_NAMES[key] ?? (key ? String(key) : 'Без типа');
const colorCase = key => CASE_COLORS[key] ?? '#8e8e93';
const verdict = record => record.review?.verdict ?? null;
const modelVerdict = record => ({confirmed:'confirmed',not_confirmed:'rejected',unknown:'uncertain'})[record.job?.analysis?.assessment] ?? null;
const inWindow = (record, start, end) => Number.isFinite(record.timestamp) && record.timestamp >= start && record.timestamp <= end;

function card(title, content, classes='') {
  return `<section class="card analytics-card ${classes}" aria-label="${esc(title)}">${content}</section>`;
}
function periodTitle(days) { return days === 1 ? 'Последние 24 часа' : `Последние ${days} дней`; }
function dateLabel(time, options={day:'numeric',month:'short'}) { return new Intl.DateTimeFormat('ru-RU', options).format(new Date(time)); }

function trend(records, start, end, days) {
  const count=days===1?24:days;
  const day0=new Date(end);day0.setHours(0,0,0,0);day0.setDate(day0.getDate()-(days-1));
  const buckets=Array.from({length:count},(_,i)=>{
    let begin,finish;
    if(days===1){begin=start+i*3600000;finish=Math.min(end,begin+3600000-1);}
    else{const d=new Date(day0);d.setDate(d.getDate()+i);begin=d.getTime();d.setDate(d.getDate()+1);finish=Math.min(end,d.getTime()-1);}
    return {start:begin,end:finish,confirmed:0,rejected:0,uncertain:0,unreviewed:0};
  });
  for(const record of records){const bucket=buckets.find(b=>inWindow(record,b.start,b.end));if(!bucket)continue;const key=verdict(record);bucket[key==='confirmed'?'confirmed':key==='rejected'?'rejected':key==='uncertain'?'uncertain':'unreviewed']++;}
  const max=Math.max(1,...buckets.map(b=>b.confirmed+b.rejected+b.uncertain+b.unreviewed));
  const bars=buckets.map(bucket=>{const total=bucket.confirmed+bucket.rejected+bucket.uncertain+bucket.unreviewed;const label=days===1?dateLabel(bucket.start,{hour:'2-digit'}):dateLabel(bucket.start,{day:'numeric'});return `<div class="analytics-trend-col" title="${esc(dateLabel(bucket.start))}: ${total} записей"><div class="analytics-trend-stack">${[['unreviewed','wait'],['uncertain','uncertain'],['rejected','rejected'],['confirmed','confirmed']].map(([key,cls])=>`<i class="analytics-seg-${cls}" style="height:${bucket[key]/max*100}%"></i>`).join('')}</div><span>${esc(label)}</span></div>`;}).join('');
  return `<div class="analytics-trend" role="img" aria-label="Распределение записей по датам и последним решениям оператора"><div class="analytics-trend-axis"><span>${max}</span><span>${Math.ceil(max/2)}</span><span>0</span></div><div class="analytics-trend-chart"><div class="analytics-gridline"></div><div class="analytics-gridline"></div><div class="analytics-gridline"></div><div class="analytics-bars">${bars}</div></div></div>`;
}

function typeSummary(records) {
  const grouped = new Map();
  for (const row of records) {
    const key = row.case ?? 'unknown';
    const item = grouped.get(key) ?? {key,total:0,analyzed:0,confirmed:0,rejected:0,uncertain:0,modelConfirmed:0,modelRejected:0,modelUnknown:0};
    item.total++;
    if (row.job?.status === 'ready' && row.job.analysis && ['confirmed', 'not_confirmed', 'unknown'].includes(row.job.analysis.assessment)) item.analyzed++;
    if (row.review?.verdict === 'confirmed') item.confirmed++;
    if (row.review?.verdict === 'rejected') item.rejected++;
    if (row.review?.verdict === 'uncertain') item.uncertain++;
    if (row.job?.analysis?.assessment === 'confirmed') item.modelConfirmed++;
    if (row.job?.analysis?.assessment === 'not_confirmed') item.modelRejected++;
    if (row.job?.analysis?.assessment === 'unknown') item.modelUnknown++;
    grouped.set(key,item);
  }
  return [...grouped.values()].sort((a,b)=>b.total-a.total || labelCase(a.key).localeCompare(labelCase(b.key),'ru'));
}

function matrixMarkup(rows, matrixMode) {
  const cols = [
    ['modelConfirmed','Модель · подтверждено'],['modelRejected','Модель · не подтверждено'],['modelUnknown','Модель · неопределённо'],
    ['confirmed','Оператор · подтвердил'],['rejected','Оператор · отклонил'],['uncertain','Оператор · неопределённо'],
  ];
  const body = rows.length ? rows.map(row => {
    const cells = cols.map(([key]) => {
      const value = row[key];
      const share = row.total ? value / row.total : 0;
      const display = matrixMode === 'share' ? pct(share * 100) : fmt(value);
      const alpha = value ? Math.min(.68, .12 + share * .56) : .04;
      return `<span class="analytics-matrix-cell" title="${esc(labelCase(row.key))}: ${fmt(value)} из ${fmt(row.total)}" style="--cell-alpha:${alpha}">${display}</span>`;
    }).join('');
    return `<div class="analytics-matrix-row"><strong>${esc(labelCase(row.key))}</strong>${cells}<b>${fmt(row.total)}</b></div>`;
  }).join('') : `<p class="analytics-empty">За выбранный период записей нет.</p>`;
  return `<div class="analytics-matrix-wrap"><div class="analytics-matrix"><div class="analytics-matrix-row analytics-matrix-head"><strong>Тип</strong>${cols.map(([,label])=>`<span>${esc(label)}</span>`).join('')}<b>Всего</b></div>${body}</div></div>`;
}

function qualityTable(rows) {
  if (!rows.length) return `<p class="analytics-empty">Для выбранного периода нет данных по типам.</p>`;
  return `<div class="analytics-table-wrap"><table class="analytics-table"><thead><tr><th>Тип события</th><th>Записей</th><th>Анализ готов</th><th>Модель: подтверд./откл./неопр.</th><th>Оператор: подтв./откл./неопр.</th><th>Совпадение</th></tr></thead><tbody>${rows.map(row=>{
    const reviewedCompared = row.records.filter(record=>verdict(record) && modelVerdict(record) && modelVerdict(record)!=='uncertain');
    const matches = reviewedCompared.filter(record=>verdict(record)===modelVerdict(record)).length;
    const agreement = reviewedCompared.length ? `${matches} / ${reviewedCompared.length} (${pct(matches / reviewedCompared.length * 100)})` : '—';
    return `<tr><th scope="row"><span class="analytics-type-dot" style="background:${colorCase(row.key)}"></span>${esc(labelCase(row.key))}</th><td>${fmt(row.total)}</td><td>${fmt(row.analyzed)}</td><td>${fmt(row.modelConfirmed)} / ${fmt(row.modelRejected)} / ${fmt(row.modelUnknown)}</td><td>${fmt(row.confirmed)} / ${fmt(row.rejected)} / ${fmt(row.uncertain)}</td><td>${agreement}</td></tr>`;
  }).join('')}</tbody></table></div><p class="analytics-note">Совпадение считается только для записей с решением оператора и определённой оценкой модели; значения модели и оператора показаны отдельно.</p>`;
}

function insights(rows, stats) {
  const items = [];
  if (stats.waiting) items.push(`<b>Ожидают решения оператора: ${fmt(stats.waiting)}.</b> Все значения взяты из готовых анализов без сохранённого отзыва.`);
  if (stats.failed) items.push(`<b>${fmt(stats.failed)} ${stats.failed===1?'обработка завершилась':'обработок завершились'} ошибкой.</b> Проверьте запись в списке источников.`);
  if (!stats.waiting && !stats.failed) items.push(`<b>Ожидающих решений нет.</b> В периоде ${fmt(stats.analyzed)} готовых анализов и ${fmt(stats.confirmed + stats.rejected + stats.uncertain)} сохранённых решений оператора.`);
  if (stats.modelUnknown) items.push(`<b>Неопределённых оценок модели: ${fmt(stats.modelUnknown)}.</b> Это отдельный результат модели; он не заменяет решение оператора.`);
  if (!items.length) items.push('<b>Недостаточно записей для выводов.</b> Здесь появятся наблюдения по мере накопления результатов анализа и проверок.');
  return `<ul class="analytics-insights">${items.slice(0,4).map(text=>`<li>${text}</li>`).join('')}</ul>`;
}


export function renderAnalytics({ records = [], periodDays = 7, now = Date.now(), matrixMode = 'count' } = {}) {
  const days = [1,7,30].includes(Number(periodDays)) ? Number(periodDays) : 7;
  const mode = matrixMode === 'share' ? 'share' : 'count';
  const end = Number.isFinite(now) ? now : Date.now();
  const day0=new Date(end);day0.setHours(0,0,0,0);day0.setDate(day0.getDate()-(days-1));
  const start=days===1?end-24*60*60*1000:day0.getTime();
  const windowRecords = records.filter(record => inWindow(record,start,end));
  const stats = summarize(windowRecords);
  const types = typeSummary(windowRecords);
  const maxType = Math.max(1,...types.map(row=>row.total));
  const typeBars = types.length ? types.map(row=>`<div class="analytics-type-row"><span class="analytics-type-name"><i style="background:${colorCase(row.key)}"></i>${esc(labelCase(row.key))}</span><span class="analytics-type-track"><i style="width:${row.total/maxType*100}%;background:${colorCase(row.key)}"></i></span><span class="analytics-type-value">${fmt(row.total)}</span></div>`).join('') : '<p class="analytics-empty">Нет записей с датой в выбранном периоде.</p>';
  const kpis = [
    ['Записи в периоде',fmt(stats.total)],['Готовые анализы',fmt(stats.analyzed)],['Ждут оператора',fmt(stats.waiting)],
    ['Оператор · подтвердил',fmt(stats.confirmed)],['Оператор · отклонил',fmt(stats.rejected)],['Модель · неопределённо',fmt(stats.modelUnknown)],
  ];
  const kpiMarkup = kpis.map(([label,value])=>`<div class="analytics-kpi"><span>${label}</span><strong>${value}</strong></div>`).join('');
  const funnel = [['Каталог',stats.total,'#8e8e93'],['Готовый анализ',stats.analyzed,'#2a78d6'],['Решение оператора',stats.confirmed+stats.rejected+stats.uncertain,'#62a384'],['Подтверждено оператором',stats.confirmed,'#487bcc']];
  const funnelMarkup = funnel.map(([label,value,color])=>`<li><i style="background:${color}"></i><span><strong>${fmt(value)}</strong><small>${label}</small></span></li>`).join('');
  const side = `<div class="analytics-mini-stats"><span><b>${stats.reviewPercent == null ? '—' : pct(stats.reviewPercent)}</b><small>записей имеют отзыв</small></span><span><b>${fmt(types.length)}</b><small>типов событий</small></span><span><b>${fmt(stats.frameCount)}</b><small>кадров разбора</small></span><span><b>${fmt(stats.objectCount)}</b><small>детекций</small></span></div>`;
  const typeRows = types.map(row=>({...row,records:windowRecords.filter(record=>(record.case ?? 'unknown')===row.key)}));
  const noDates = records.filter(record=>!Number.isFinite(record.timestamp)).length;
  const dateNote = noDates ? `<p class="analytics-note">${fmt(noDates)} записей каталога не имеют даты анализа и не включены в период.</p>` : '';
  return `<div class="analytics-page">
    <div class="analytics-heading"><div><h1>Аналитика</h1><p>${esc(periodTitle(days))} · Записей с датой: ${fmt(stats.total)}</p></div><div class="analytics-segmented" role="group" aria-label="Период">${[[1,'24 ч'],[7,'7 дней'],[30,'30 дней']].map(([value,label])=>`<button class="btn" type="button" data-period="${value}" aria-pressed="${days===value}">${label}</button>`).join('')}</div></div>
    ${dateNote}
    <section class="analytics-kpis" aria-label="Ключевые показатели">${kpiMarkup}</section>
    <div class="analytics-grid analytics-grid-wide">
      ${card('Динамика записей',`<div class="analytics-card-head"><div><h2>Динамика записей</h2><p>Решения оператора по дате анализа</p></div><div class="analytics-legend"><span><i class="analytics-seg-confirmed"></i>Подтвердил</span><span><i class="analytics-seg-rejected"></i>Отклонил</span><span><i class="analytics-seg-uncertain"></i>Неопределённо</span><span><i class="analytics-seg-wait"></i>Без решения</span></div></div>${stats.total ? trend(windowRecords,start,end,days) : '<p class="analytics-empty">Нет записей с датой в выбранном периоде.</p>'}`,'analytics-trend-card')}
      ${card('Воронка решений',`<div class="analytics-card-head"><div><h2>Воронка решений</h2><p>Фактическое число записей</p></div></div><ol class="analytics-funnel">${funnelMarkup}</ol>${side}`,'analytics-funnel-card')}
    </div>
    <div class="analytics-grid">
      ${card('Записи по типу',`<div class="analytics-card-head"><div><h2>Записи по типу</h2><p>Число каталогизированных примеров</p></div></div><div class="analytics-type-bars">${typeBars}</div>`,'analytics-types-card')}
      ${card('Что требует внимания',`<div class="analytics-card-head"><div><h2>Что требует внимания</h2><p>Наблюдения из данных периода</p></div></div>${insights(types,stats)}`,'analytics-insights-card')}
    </div>
    ${card('Матрица решений',`<div class="analytics-card-head"><div><h2>Матрица решений</h2><p>Оценки модели и решения оператора по типу</p></div><div class="analytics-segmented" role="group" aria-label="Представление матрицы"><button class="btn" type="button" data-matrix="count" aria-pressed="${mode==='count'}">Число</button><button class="btn" type="button" data-matrix="share" aria-pressed="${mode==='share'}">Доля</button></div></div>${matrixMarkup(types,mode)}`,'analytics-matrix-card')}
    ${card('Качество по типам',`<div class="analytics-card-head"><div><h2>Сопоставление решений по типу</h2><p>Сравнение только там, где есть обе оценки</p></div></div>${qualityTable(typeRows)}`,'analytics-quality-card')}
  </div>`;
}
