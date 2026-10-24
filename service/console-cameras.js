import { objectNames } from './viewer-annotations.js';

const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const caseLabels = { fall: 'Падение', phone: 'Телефон', smoking: 'Курение', proximity: 'Техника' };
const safeUrl = value => /^https?:\/\//i.test(String(value || '')) ? esc(value) : '';
const formatTime = value => {
  if (!value) return 'Время не указано';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat('ru-RU', { dateStyle: 'medium', timeStyle: 'short' }).format(date);
};
const processedCV = record => record?.job?.status === 'ready' && Boolean(record.job.result || record.job.analysis);
const latestFrame = record => record?.job?.result?.frames?.[0] || (processedCV(record) ? record.job?.analysis?.frames?.[0] : null) || null;
const reviewLabel = record => {
  if (!record?.job) return 'Не анализировалось';
  if (record.job.status === 'failed') return 'Разбор не завершён';
  if (record.job.analysisError) return 'Разбор не завершён';
  if (['detecting','analyzing'].includes(record.job.status)) return latestFrame(record) ? 'Видео с рамками доступно' : 'Обработка видео';
  if (!processedCV(record)) return 'Ожидает разбора';
  if (record.review) return 'Проверено оператором';
  if (record.job.result?.episode) return 'Ожидает решения';
  if (record.job.analysis || record.job.result) return 'Кадры разобраны';
  return 'Ожидает разбора';
};
const mediaLabel = sample => sample?.video ? 'Видео' : 'Кадр';
const counts = record => {
  const detections = latestFrame(record)?.detections || [];
  if (!latestFrame(record)) return { people: null, objects: null, classes: [] };
  const people = detections.filter(item => item.label === 'person').length;
  const objects = detections.filter(item => item.label !== 'person').length;
  const classes = [...new Set(detections.filter(item => item.label !== 'person').map(item => item.label).filter(Boolean))];
  return { people, objects, classes };
};
function sourceTile(record, selectedId) {
  const { sample } = record;
  const id = String(sample.id);
  const selected = id === selectedId;
  const img = sample.image || sample.frames?.[0] || '';
  const review = reviewLabel(record);
  const analyzed = processedCV(record);
  return `<button class="camera-tile ${selected ? 'is-selected' : ''}" type="button" data-camera-source="${esc(id)}" aria-pressed="${selected}" aria-label="Открыть источник ${esc(sample.title || id)}">
    <span class="camera-tile-image">${img ? `<img src="${esc(img)}" alt="" loading="lazy">` : '<span class="camera-no-image">Нет изображения</span>'}<span class="camera-tile-type">${sample.video ? 'Видео' : 'Кадр'}</span></span>
    <span class="camera-tile-content"><span class="camera-tile-title">${esc(sample.title || id)}</span><span class="camera-tile-meta">${esc(sample.source || 'Источник не указан')} · ${esc(caseLabels[sample.case] || sample.case || 'Без сценария')}</span><span class="camera-tile-foot"><span>${analyzed ? 'Разобрано' : latestFrame(record) ? 'Рамки доступны' : 'Без разбора'}</span><span>${esc(review)}</span></span></span>
  </button>`;
}

export function renderCameras({ records = [], selectedId, viewerMarkup = '', caseFilter = 'all', sort = 'catalog' } = {}) {
  const all = Array.isArray(records) ? records.filter(record => record?.sample?.id) : [];
  if (!all.length) return `<section class="camera-console"><div class="camera-page-heading"><div><h1>Камеры</h1><p>Видеозаписи и кадры для разбора и проверки.</p></div></div><div class="camera-empty"><h2>Источники не добавлены</h2><p>Когда появятся видеозаписи или кадры, они будут доступны здесь.</p></div></section>`;

  const selected = all.find(record => String(record.sample.id) === String(selectedId)) || all[0];
  const activeId = String(selected.sample.id);
  const filtered = all.filter(record => caseFilter === 'all' || record.sample.case === caseFilter);
  const sorted = filtered.slice();
  if (sort === 'recent') sorted.sort((a, b) => new Date(b.timestamp || b.job?.createdAt || b.sample.timestamp || 0) - new Date(a.timestamp || a.job?.createdAt || a.sample.timestamp || 0));
  if (sort === 'objects') sorted.sort((a, b) => counts(b).objects - counts(a).objects || String(a.sample.title || '').localeCompare(String(b.sample.title || ''), 'ru'));
  const visible = sorted;
  const videos = all.filter(record => Boolean(record.sample.video)).length;
  const images = all.length - videos;
  const analyzed = all.filter(processedCV).length;
  const awaiting = all.filter(record => processedCV(record) && record.job?.result?.episode && !record.review).length;
  const { people, objects, classes } = counts(selected);
  const sample = selected.sample;
  const url = safeUrl(sample.sourceUrl);
  const title = sample.title || sample.id;
  const action = selected.job?.analysis?.action || selected.job?.analysis?.summary || sample.rule || 'Для этого источника пока нет текстового описания.';
  const classMarkup = classes.length ? classes.map(label => `<span class="camera-class-chip">${esc(objectNames[label] || label)}</span>`).join('') : `<span class="camera-context-empty">${latestFrame(selected) ? 'Нет объектов в выбранном кадре' : 'Разбор пока недоступен'}</span>`;
  const licenseUrl = safeUrl(sample.licenseUrl || sample.licenseURL);
  const license = sample.license || sample.licenseName || '';
  const countLabel = value => value == null ? '—' : value;
  const countContext = latestFrame(selected) ? 'В первом кадре разбора' : 'Разбор пока недоступен';

  return `<section class="camera-console">
    <div class="camera-page-heading"><div><h1>Камеры</h1><p>Видеозаписи и кадры · ${all.length} источника</p></div></div>
    <div class="camera-summary" aria-label="Сводка источников">
      <span><b>${videos}</b> видео</span><span><b>${images}</b> кадров</span><span><b>${analyzed}</b> разобрано</span><span><b>${awaiting}</b> ждут решения</span>
    </div>
    <div class="camera-controls-row">
      <div class="camera-filter-group" role="group" aria-label="Сценарий">
        ${[['all','Все'],['fall','Падение'],['phone','Телефон'],['smoking','Курение'],['proximity','Техника']].map(([id,label]) => `<button type="button" class="camera-filter ${caseFilter === id ? 'active' : ''}" data-camera-case="${id}" aria-pressed="${caseFilter === id}">${label}</button>`).join('')}
      </div>
      <div class="camera-filter-group camera-sort-group" role="group" aria-label="Сортировка">
        ${[['catalog','По каталогу'],['objects','По объектам'],['recent','Сначала новые']].map(([id,label]) => `<button type="button" class="camera-filter ${sort === id ? 'active' : ''}" data-camera-sort="${id}" aria-pressed="${sort === id}">${label}</button>`).join('')}
      </div>
    </div>
    <section class="camera-selected" aria-label="Выбранный источник">
      <div class="camera-viewer">${viewerMarkup}</div>
      <div class="camera-selected-info">
        <div class="camera-selected-title"><span class="camera-kind">${mediaLabel(sample)}</span><h2>${esc(title)}</h2><p>${esc(caseLabels[sample.case] || sample.case || 'Сценарий не указан')} · ${esc(formatTime(selected.timestamp || selected.job?.createdAt || sample.timestamp))}</p></div>
        <p class="camera-context-copy">${esc(action)}</p>
        <div class="camera-counts" aria-label="${esc(countContext)}"><div><b>${countLabel(people)}</b><span>людей в кадре</span></div><div><b>${countLabel(objects)}</b><span>объектов</span></div></div><p class="camera-count-note">${esc(countContext)}</p>
        <div class="camera-classes"><h3>Объекты в кадре</h3><div>${classMarkup}</div></div>
        <div class="camera-provenance"><span>Источник материала</span>${url ? `<a href="${url}" target="_blank" rel="noreferrer">${esc(sample.source || 'Открыть источник')} <span aria-hidden="true">↗</span></a>` : `<span>${esc(sample.source || 'Не указан')}</span>`}<small>${esc(sample.originalFile || sample.id)}</small>${license ? `<small>${licenseUrl ? `<a href="${licenseUrl}" target="_blank" rel="noreferrer">Лицензия: ${esc(license)}</a>` : `Лицензия: ${esc(license)}`}</small>` : ''}</div>
        <button class="btn primary camera-open-events" type="button" data-source="${esc(sample.id)}">Открыть связанные события</button>
      </div>
    </section>
    <section class="camera-catalog" aria-label="Все источники">
      <div class="camera-catalog-heading"><h2>Источники</h2><span>${visible.length} из ${all.length}</span></div>
      <div class="camera-tile-grid">${visible.map(record => sourceTile(record, activeId)).join('') || '<p class="camera-empty-inline">Нет источников для выбранного сценария.</p>'}</div>
    </section>
  </section>`;
}
