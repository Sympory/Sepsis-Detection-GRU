/**
 * SepsisAI — YBÜ Karar Destek İstemcisi
 * =====================================
 * Hasta listesi, saatlik veri girişi, risk + güvenilirlik paneli,
 * risk trendi (belirsizlik bandı ile), model & kalibrasyon sayfası.
 */

const API_BASE = '';
const MIN_CORE_VITALS = 3;

let currentPatientId = null;
let currentPatient = null;
let currentHourly = [];
let allPatients = [];
let riskChart = null;
let modelInfo = null;

// ===================== FORM ŞEMASI =====================
// range: klinik normal aralık (yer tutucu ve uyarı için), hard: sunucunun kabul ettiği sınırlar
const FIELD_GROUPS = [
    {
        id: 'core', title: 'Temel vital bulgular', open: true, tag: 'Zorunlu (en az 3)',
        note: 'Model bu değerler olmadan hastayı "ortalama/normal" kabul eder. HR, SpO2, ateş, solunum ve MAP\'ten en az üçünü girin.',
        fields: [
            { id: 'HR', label: 'Nabız', unit: 'atım/dk', range: [60, 100], hard: [40, 200], core: true },
            { id: 'O2Sat', label: 'SpO2', unit: '%', range: [95, 100], hard: [70, 100], core: true },
            { id: 'Temp', label: 'Ateş', unit: '°C', range: [36, 37.5], hard: [35, 42], step: 0.1, core: true },
            { id: 'Resp', label: 'Solunum', unit: '/dk', range: [12, 20], hard: [8, 50], core: true },
            { id: 'MAP', label: 'MAP', unit: 'mmHg', range: [70, 100], hard: [40, 180], core: true, help: 'Boşsa SBP/DBP\'den hesaplanır' },
            { id: 'SBP', label: 'Sistolik KB', unit: 'mmHg', range: [100, 140], hard: [60, 250] },
            { id: 'DBP', label: 'Diyastolik KB', unit: 'mmHg', range: [60, 90], hard: [30, 150] },
        ],
    },
    {
        id: 'keylabs', title: 'Kritik laboratuvar', open: true, tag: 'Önerilen',
        note: 'Sepsis ve organ disfonksiyonu için en bilgilendirici değerler. Son 6 saatte ölçüldüyse girin.',
        fields: [
            { id: 'Lactate', label: 'Laktat', unit: 'mmol/L', range: [0.5, 2], hard: [0.5, 20], step: 0.1 },
            { id: 'WBC', label: 'WBC', unit: '10³/µL', range: [4, 12], hard: [1, 50], step: 0.1 },
            { id: 'Platelets', label: 'Trombosit', unit: '10³/µL', range: [150, 400], hard: [20, 800] },
            { id: 'Creatinine', label: 'Kreatinin', unit: 'mg/dL', range: [0.6, 1.2], hard: [0.3, 15], step: 0.01 },
            { id: 'Bilirubin_total', label: 'Total bilirubin', unit: 'mg/dL', range: [0.1, 1.2], hard: [0.1, 30], step: 0.1 },
            { id: 'pH', label: 'pH', unit: '', range: [7.35, 7.45], hard: [6.8, 7.8], step: 0.01 },
        ],
    },
    {
        id: 'resp', title: 'Solunum ve kan gazı', open: false,
        fields: [
            { id: 'FiO2', label: 'FiO2', unit: 'oran', range: [0.21, 0.5], hard: [0.21, 100], step: 0.01, help: '0.21–1.0 (ya da % girin, çevrilir)' },
            { id: 'SaO2', label: 'SaO2', unit: '%', range: [95, 100], hard: [50, 100] },
            { id: 'PaCO2', label: 'PaCO2', unit: 'mmHg', range: [35, 45], hard: [15, 100] },
            { id: 'HCO3', label: 'HCO3', unit: 'mmol/L', range: [22, 26], hard: [10, 45], step: 0.1 },
            { id: 'BaseExcess', label: 'Baz açığı', unit: 'mmol/L', range: [-2, 2], hard: [-20, 20], step: 0.1 },
            { id: 'EtCO2', label: 'EtCO2', unit: 'mmHg', range: [35, 45], hard: [10, 80] },
        ],
    },
    {
        id: 'chem', title: 'Biyokimya', open: false,
        fields: [
            { id: 'BUN', label: 'BUN', unit: 'mg/dL', range: [7, 20], hard: [3, 150] },
            { id: 'Glucose', label: 'Glukoz', unit: 'mg/dL', range: [70, 140], hard: [30, 600] },
            { id: 'AST', label: 'AST', unit: 'U/L', range: [10, 40], hard: [5, 5000] },
            { id: 'Alkalinephos', label: 'ALP', unit: 'U/L', range: [40, 130], hard: [10, 2000] },
            { id: 'Bilirubin_direct', label: 'Direkt bilirubin', unit: 'mg/dL', range: [0, 0.3], hard: [0, 15], step: 0.1 },
            { id: 'Potassium', label: 'Potasyum', unit: 'mmol/L', range: [3.5, 5], hard: [2, 8], step: 0.1 },
            { id: 'Chloride', label: 'Klor', unit: 'mmol/L', range: [96, 106], hard: [70, 130] },
            { id: 'Calcium', label: 'Kalsiyum', unit: 'mg/dL', range: [8.5, 10.5], hard: [5, 15], step: 0.1 },
            { id: 'Magnesium', label: 'Magnezyum', unit: 'mg/dL', range: [1.7, 2.4], hard: [0.5, 5], step: 0.1 },
            { id: 'Phosphate', label: 'Fosfat', unit: 'mg/dL', range: [2.5, 4.5], hard: [0.5, 15], step: 0.1 },
        ],
    },
    {
        id: 'heme', title: 'Hematoloji ve pıhtılaşma', open: false,
        fields: [
            { id: 'Hgb', label: 'Hemoglobin', unit: 'g/dL', range: [12, 17], hard: [5, 20], step: 0.1 },
            { id: 'Hct', label: 'Hematokrit', unit: '%', range: [36, 50], hard: [15, 65], step: 0.1 },
            { id: 'PTT', label: 'PTT', unit: 'sn', range: [25, 35], hard: [15, 150], step: 0.1 },
            { id: 'Fibrinogen', label: 'Fibrinojen', unit: 'mg/dL', range: [200, 400], hard: [30, 1200] },
            { id: 'TroponinI', label: 'Troponin I', unit: 'ng/mL', range: [0, 0.04], hard: [0, 100], step: 0.01 },
        ],
    },
    {
        id: 'bio', title: 'Biyobelirteçler', open: false, tag: 'Modelde kullanılmaz', tagOff: true,
        note: 'Bu değerler eğitim verisinde güvenilir olmadığı için modelden çıkarıldı. Yalnızca kayıt amaçlı saklanır, risk skorunu etkilemez.',
        fields: [
            { id: 'PCT', label: 'Prokalsitonin', unit: 'ng/mL', range: [0, 0.5], hard: [0.01, 100], step: 0.01 },
            { id: 'CRP', label: 'CRP', unit: 'mg/L', range: [0, 10], hard: [0, 500], step: 0.1 },
            { id: 'Presepsin', label: 'Presepsin', unit: 'pg/mL', range: [0, 600], hard: [100, 5000] },
            { id: 'IL6', label: 'IL-6', unit: 'pg/mL', range: [0, 7], hard: [0, 1000], step: 0.1 },
        ],
    },
];

const ALL_FIELDS = FIELD_GROUPS.flatMap(g => g.fields);
const FIELD_BY_ID = Object.fromEntries(ALL_FIELDS.map(f => [f.id, f]));
const CORE_IDS = ALL_FIELDS.filter(f => f.core).map(f => f.id);

const COMPONENT_LABELS = {
    completeness: 'Veri tamlığı',
    temporal: 'Zaman kapsamı',
    certainty: 'Model kararlılığı',
    in_distribution: 'Dağılım uyumu',
};

// ===================== INIT =====================
document.addEventListener('DOMContentLoaded', () => {
    renderFieldGroups();
    startClock();
    checkHealth();
    loadPatients();
    loadModelInfo();
    setInterval(loadPatients, 60000);

    document.getElementById('modal-add-patient').addEventListener('click', e => {
        if (e.target.id === 'modal-add-patient') closeAddPatientModal();
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAddPatientModal(); });
});

function startClock() {
    const el = document.getElementById('clock');
    const tick = () => { el.textContent = new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }); };
    tick();
    setInterval(tick, 15000);
}

function toggleTheme() {
    const root = document.documentElement;
    const isDark = root.dataset.theme === 'dark' ||
        (!root.dataset.theme && window.matchMedia('(prefers-color-scheme: dark)').matches);
    root.dataset.theme = isDark ? 'light' : 'dark';
    try { localStorage.setItem('sepsis-theme', root.dataset.theme); } catch (e) { }
    if (currentHourly.length) renderRiskChart(currentHourly);
}

async function checkHealth() {
    const box = document.getElementById('model-status');
    const text = document.getElementById('model-status-text');
    try {
        const r = await fetch(`${API_BASE}/api/health`);
        const d = await r.json();
        const ok = d.model_loaded && d.preprocessing_loaded;
        box.className = `status ${ok ? 'ok' : 'err'}`;
        text.textContent = ok ? (d.calibrator_loaded ? 'Model + kalibrasyon aktif' : 'Model aktif (kalibrasyon yok)') : 'Model yüklenemedi';
    } catch (e) {
        box.className = 'status err';
        text.textContent = 'Sunucuya ulaşılamıyor';
    }
}

// ===================== NAVIGATION =====================
const VIEW_TITLES = {
    'dashboard': ['Genel Bakış', 'Ünitedeki hastaların güncel sepsis riski'],
    'patients': ['Hastalar', 'Kayıtlı tüm hastalar'],
    'patient-detail': ['Hasta', 'Saatlik veri girişi, risk ve güvenilirlik'],
    'alerts': ['Alarmlar', 'Yüksek riskli ve değerlendirilemeyen hastalar'],
    'model-info': ['Model & Güvenilirlik', 'Aktif modelin ölçülmüş performansı ve kalibrasyonu'],
};

function switchView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${name}`));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === name));
    const [title, sub] = VIEW_TITLES[name] || ['', ''];
    document.getElementById('page-title').textContent = title;
    document.getElementById('page-subtitle').textContent = sub;
    document.getElementById('sidebar').classList.remove('open');
    window.scrollTo({ top: 0 });
}

// ===================== HELPERS =====================
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function genderText(g) { return g === 'Male' ? 'Erkek' : g === 'Female' ? 'Kadın' : '—'; }

function riskClass(risk) {
    if (!risk) return 'none';
    const r = risk.toLocaleLowerCase('tr-TR');
    if (r.includes('yetersiz')) return 'none';
    if (r.includes('çok yüksek') || r.includes('kritik')) return 'critical';
    if (r.includes('yüksek')) return 'high';
    if (r.includes('orta')) return 'moderate';
    if (r.includes('çok düşük')) return 'very-low';
    if (r.includes('düşük')) return 'low';
    return 'none';
}

const RISK_ORDER = { 'critical': 5, 'high': 4, 'moderate': 3, 'low': 2, 'very-low': 1, 'none': 0 };

function riskBadge(risk) {
    if (!risk) return '<span class="badge none">Veri yok</span>';
    return `<span class="badge ${riskClass(risk)}">${escapeHtml(risk)}</span>`;
}

function relClass(score) { return score >= 70 ? 'high' : score >= 45 ? 'mid' : 'low'; }

function relBadge(score, insufficient) {
    if (score === null || score === undefined) return '<span class="muted small">—</span>';
    const cls = insufficient ? 'low' : relClass(score);
    return `<span class="rel ${cls}"><span class="rel-meter"><i style="width:${Math.max(4, score)}%"></i></span>${Math.round(score)}</span>`;
}

function pct(v, digits = 1) { return v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`; }

function avatarHtml(p) {
    return `<div class="avatar">${escapeHtml((p.name || '?').trim().charAt(0).toLocaleUpperCase('tr-TR'))}</div>`;
}

function timeAgo(ts) {
    if (!ts) return '—';
    const d = new Date(ts.replace(' ', 'T') + (ts.includes('Z') || ts.includes('+') ? '' : 'Z'));
    const mins = Math.round((Date.now() - d.getTime()) / 60000);
    if (isNaN(mins)) return '—';
    if (mins < 1) return 'şimdi';
    if (mins < 60) return `${mins} dk önce`;
    if (mins < 1440) return `${Math.round(mins / 60)} sa önce`;
    return d.toLocaleDateString('tr-TR');
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// ===================== PATIENTS =====================
async function loadPatients() {
    try {
        const r = await fetch(`${API_BASE}/api/patients`);
        const d = await r.json();
        if (!d.success) throw new Error(d.error);
        allPatients = d.patients || [];
        renderDashboard();
        renderAllPatients();
        renderAlerts();
    } catch (e) {
        console.error(e);
        showToast('Hasta verileri yüklenemedi', 'error');
    }
}

function sortedByRisk(list) {
    return [...list].sort((a, b) =>
        (RISK_ORDER[riskClass(b.latest_risk)] - RISK_ORDER[riskClass(a.latest_risk)]) ||
        ((b.latest_prediction ?? -1) - (a.latest_prediction ?? -1)));
}

function renderDashboard() {
    const cls = allPatients.map(p => riskClass(p.latest_risk));
    document.getElementById('stat-total').textContent = allPatients.length;
    document.getElementById('stat-high-risk').textContent = cls.filter(c => c === 'high' || c === 'critical').length;
    document.getElementById('stat-moderate').textContent = cls.filter(c => c === 'moderate').length;
    document.getElementById('stat-unassessed').textContent = cls.filter(c => c === 'none').length;

    const tbody = document.getElementById('dashboard-patients-body');
    if (!allPatients.length) {
        tbody.innerHTML = `<tr><td colspan="5"><div class="empty"><h3>Henüz hasta yok</h3><p>Sağ üstteki “Yeni Hasta” ile başlayın.</p></div></td></tr>`;
        return;
    }
    tbody.innerHTML = sortedByRisk(allPatients).slice(0, 10).map(p => `
        <tr class="clickable" onclick="openPatient(${p.id})">
            <td><div class="person">${avatarHtml(p)}<div><strong>${escapeHtml(p.name)}</strong><span>${escapeHtml(p.patient_id)} · ${p.age || '—'} yaş</span></div></div></td>
            <td>${riskBadge(p.latest_risk)}</td>
            <td class="num">${riskClass(p.latest_risk) === 'none' ? '—' : pct(p.latest_prediction)}</td>
            <td>${relBadge(p.latest_reliability, riskClass(p.latest_risk) === 'none')}</td>
            <td class="muted small">${p.latest_hour != null ? `Saat ${p.latest_hour} · ${timeAgo(p.latest_timestamp)}` : 'Ölçüm yok'}</td>
        </tr>`).join('');
}

function renderAllPatients() {
    const q = (document.getElementById('patient-search').value || '').toLocaleLowerCase('tr-TR');
    const list = allPatients.filter(p => !q ||
        (p.name || '').toLocaleLowerCase('tr-TR').includes(q) || (p.patient_id || '').toLocaleLowerCase('tr-TR').includes(q));
    const tbody = document.getElementById('all-patients-body');
    if (!list.length) {
        tbody.innerHTML = `<tr><td colspan="8"><div class="empty"><h3>${allPatients.length ? 'Eşleşen hasta yok' : 'Henüz hasta yok'}</h3></div></td></tr>`;
        return;
    }
    tbody.innerHTML = list.map(p => `
        <tr class="clickable" onclick="openPatient(${p.id})">
            <td><div class="person">${avatarHtml(p)}<div><strong>${escapeHtml(p.name)}</strong></div></div></td>
            <td class="num">${escapeHtml(p.patient_id)}</td>
            <td>${p.age || '—'}</td>
            <td>${genderText(p.gender)}</td>
            <td>${riskBadge(p.latest_risk)}</td>
            <td>${relBadge(p.latest_reliability, riskClass(p.latest_risk) === 'none')}</td>
            <td class="num">${p.hourly_data_count || 0}</td>
            <td><button class="btn btn-danger-ghost btn-sm" onclick="event.stopPropagation(); deletePatient(${p.id})">Sil</button></td>
        </tr>`).join('');
}

function renderAlerts() {
    const high = sortedByRisk(allPatients.filter(p => ['high', 'critical'].includes(riskClass(p.latest_risk))));
    const insufficient = allPatients.filter(p => (p.latest_risk || '').includes('Yetersiz'));
    const lowRel = allPatients.filter(p => !['none'].includes(riskClass(p.latest_risk)) && p.latest_reliability != null && p.latest_reliability < 45);

    const badge = document.getElementById('alert-count');
    badge.hidden = high.length === 0;
    badge.textContent = high.length;

    const banner = document.getElementById('critical-alert');
    banner.hidden = high.length === 0;
    document.getElementById('critical-alert-text').textContent = `${high.length} hasta yüksek veya çok yüksek risk seviyesinde.`;

    const row = (p, extra) => `
        <div class="alert-row" onclick="openPatient(${p.id})">
            <div class="person">${avatarHtml(p)}<div><strong>${escapeHtml(p.name)}</strong><span>${escapeHtml(p.patient_id)} · Saat ${p.latest_hour ?? '—'}</span></div></div>
            ${extra}
        </div>`;

    let html = '';
    if (high.length) html += high.map(p => row(p, `${riskBadge(p.latest_risk)}<span class="num">${pct(p.latest_prediction)}</span>${relBadge(p.latest_reliability)}`)).join('');
    if (insufficient.length) html += insufficient.map(p => row(p, `<span class="badge none">Yetersiz veri — tekrar ölçüm girin</span>`)).join('');
    if (lowRel.length) html += lowRel.map(p => row(p, `${riskBadge(p.latest_risk)}<span class="rel low">Düşük güvenilirlik</span>`)).join('');
    document.getElementById('alerts-body').innerHTML = html ||
        `<div class="empty"><h3>Aktif alarm yok</h3><p>Yüksek riskli veya değerlendirilemeyen hastalar burada listelenir.</p></div>`;
}

// ===================== PATIENT DETAIL =====================
async function openPatient(id) {
    currentPatientId = id;
    const nav = document.getElementById('nav-patient-detail');
    nav.hidden = false;
    switchView('patient-detail');

    try {
        const r = await fetch(`${API_BASE}/api/patients/${id}`);
        const d = await r.json();
        if (!d.success) throw new Error(d.error);

        currentPatient = d.patient;
        currentHourly = d.hourly_data || [];

        document.getElementById('nav-patient-name').textContent = currentPatient.name;
        document.getElementById('detail-name').textContent = currentPatient.name;
        document.getElementById('detail-meta').textContent =
            `ID ${currentPatient.patient_id} · ${currentPatient.age ? currentPatient.age + ' yaş' : 'yaş girilmemiş'} · ${genderText(currentPatient.gender)} · ${currentHourly.length} saatlik kayıt`;
        document.getElementById('detail-avatar').textContent = (currentPatient.name || '?').trim().charAt(0).toLocaleUpperCase('tr-TR');

        resetEntryForm();
        renderHourlyHistory(currentHourly);
        renderRiskChart(currentHourly);

        const last = currentHourly[currentHourly.length - 1];
        if (last && last.assessment) {
            renderAssessment({ prediction: last.prediction, risk_level: last.risk_level, hour: last.hour, reliability: last.assessment });
        } else if (last) {
            renderAssessment({ prediction: last.prediction, risk_level: last.risk_level, hour: last.hour, reliability: null });
        } else {
            document.getElementById('assess-empty').hidden = false;
            document.getElementById('assess-body').hidden = true;
        }
    } catch (e) {
        console.error(e);
        showToast('Hasta bilgileri yüklenemedi', 'error');
    }
}

function resetEntryForm() {
    document.getElementById('hourly-data-form').reset();
    const last = currentHourly[currentHourly.length - 1];
    document.getElementById('inp-Hour').value = last ? last.hour + 1 : 1;
    document.getElementById('btn-copy-last').disabled = !last;
    ALL_FIELDS.forEach(f => validateField(f.id));
    updateCompleteness();
}

function copyLastVitals() {
    const last = currentHourly[currentHourly.length - 1];
    if (!last) return;
    const vs = last.vital_signs || {};
    ALL_FIELDS.forEach(f => {
        const el = document.getElementById(`inp-${f.id}`);
        // Yalnızca vital bulgular kopyalanır; laboratuvar değerleri her saat tekrar ölçülmez
        if (f.core || ['SBP', 'DBP'].includes(f.id)) el.value = vs[f.id] ?? '';
        validateField(f.id);
    });
    updateCompleteness();
    showToast('Son saatin vital bulguları kopyalandı — değerleri kontrol edip güncelleyin', 'info');
}

function renderHourlyHistory(rows) {
    const tbody = document.getElementById('hourly-history');
    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="6"><div class="empty"><h3>Henüz kayıt yok</h3><p>Yukarıdaki formla ilk saatlik veriyi girin.</p></div></td></tr>`;
        return;
    }
    tbody.innerHTML = [...rows].reverse().map(e => {
        const v = e.vital_signs || {};
        const chips = [['HR', 'HR'], ['O2Sat', 'SpO2'], ['Temp', 'T'], ['Resp', 'SS'], ['MAP', 'MAP'], ['Lactate', 'Lak'], ['WBC', 'WBC']]
            .filter(([k]) => v[k] !== undefined && v[k] !== null)
            .map(([k, l]) => `<span class="chip"><b>${l}</b>${escapeHtml(v[k])}</span>`).join('');
        const insufficient = e.assessment ? !e.assessment.sufficient_data : false;
        return `
            <tr>
                <td class="num">${e.hour}</td>
                <td><div class="vitals">${chips || '<span class="muted">—</span>'}</div></td>
                <td class="num">${insufficient ? '—' : pct(e.prediction)}</td>
                <td>${riskBadge(e.risk_level)}</td>
                <td>${relBadge(e.reliability, insufficient)}</td>
                <td>${e.assessment ? `<button class="btn btn-ghost btn-sm" onclick="showHistoricAssessment(${e.hour})">Detay</button>` : ''}</td>
            </tr>`;
    }).join('');
}

function showHistoricAssessment(hour) {
    const e = currentHourly.find(r => r.hour === hour);
    if (!e) return;
    renderAssessment({ prediction: e.prediction, risk_level: e.risk_level, hour: e.hour, reliability: e.assessment });
    document.getElementById('assessment').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===================== ASSESSMENT PANEL =====================
function renderAssessment(res) {
    const rel = res.reliability;
    const threshold = modelInfo ? modelInfo.threshold : 0.24;
    const insufficient = rel ? !rel.sufficient_data : false;
    const cls = riskClass(res.risk_level);
    const color = `var(--risk-${cls === 'none' ? 'none' : cls})`;

    document.getElementById('assess-empty').hidden = true;
    const body = document.getElementById('assess-body');
    body.hidden = false;

    // --- Risk bölümü
    let riskHtml;
    if (insufficient) {
        riskHtml = `
            <div class="assess-section">
                <div class="assess-kicker"><span>Sepsis riski · Saat ${res.hour}</span></div>
                <div class="risk-main">
                    <div class="risk-score" style="color:${color}">—</div>
                    <div class="risk-label" style="color:${color}">Değerlendirilemedi</div>
                </div>
                <div class="risk-sub">Ham model çıktısı ${pct(res.prediction)} — yetersiz veriyle üretildiği için risk sınıfı verilmedi.</div>
            </div>`;
    } else {
        const mc = rel && rel.mc_dropout;
        const pos = v => `${Math.min(100, Math.max(0, v * 100))}%`;
        riskHtml = `
            <div class="assess-section">
                <div class="assess-kicker"><span>Model skoru · Saat ${res.hour}</span><span>eşik ${pct(threshold, 0)}</span></div>
                <div class="risk-main">
                    <div class="risk-score" style="color:${rel && rel.clinical_disagreement ? 'var(--text-2)' : color}">${(res.prediction * 100).toFixed(1)}<small>%</small></div>
                    <div class="risk-label" style="color:${color}">${escapeHtml(res.risk_level)}
                        ${rel && rel.clinical_disagreement ? '<div class="risk-label-sub">klinik kurallarla yükseltildi</div>' : ''}</div>
                </div>
                <div class="scale">
                    <div class="scale-threshold" style="left:${pos(threshold)}"></div>
                    ${mc ? `<div class="scale-band" style="left:${pos(mc.p05)}; width:calc(${pos(mc.p95)} - ${pos(mc.p05)})"></div>` : ''}
                    <div class="scale-marker" style="left:${pos(res.prediction)}"></div>
                </div>
                <div class="risk-sub">
                    ${mc ? `Belirsizlik aralığı (%90): <b>${pct(mc.p05, 0)} – ${pct(mc.p95, 0)}</b><br>` : ''}
                    ${rel && rel.calibrated_probability != null ? `Kalibre olasılık: <b>${pct(rel.calibrated_probability)}</b> — bu skordaki test hastalarının gözlenen sepsis oranı` : ''}
                </div>
            </div>`;
    }

    // --- Güvenilirlik bölümü
    let relHtml = '';
    if (rel) {
        const score = rel.score;
        const rc = insufficient ? 'low' : relClass(score);
        const ringColor = `var(--rel-${rc})`;
        const desc = insufficient ? 'Temel vital bulgular eksik'
            : rc === 'high' ? 'Sonuç yeterli veriye ve kararlı bir model çıktısına dayanıyor'
            : rc === 'mid' ? 'Sonucu klinik değerlendirmeyle birlikte yorumlayın'
            : 'Sonuca dayanarak karar vermeyin';
        relHtml = `
            <div class="assess-section">
                <div class="assess-kicker"><span>Tahmin güvenilirliği</span></div>
                <div class="rel-main">
                    <div class="rel-ring" style="--p:${score}; --c:${ringColor}"><span>${Math.round(score)}</span></div>
                    <div>
                        <div class="rel-title" style="color:${ringColor}">${escapeHtml(rel.level)}</div>
                        <div class="rel-desc">${desc}</div>
                    </div>
                </div>
                <div class="components">
                    ${Object.keys(COMPONENT_LABELS).filter(k => k in rel.components).map(k => [k, rel.components[k]]).map(([k, v]) => `
                        <div class="component"><span>${COMPONENT_LABELS[k] || k}</span>
                        <div class="bar"><div class="bar-fill" style="width:${v}%"></div></div><b>${Math.round(v)}</b></div>`).join('')}
                </div>
            </div>`;

        const items = [];
        (rel.red_flags || []).forEach(f => items.push(`<li class="flag">Klinik bayrak: ${escapeHtml(FIELD_BY_ID[f.feature]?.label || f.feature)} = ${escapeHtml(f.value)}</li>`));
        (rel.reasons || []).forEach((r, i) => items.push(`<li class="${i === 0 && rel.clinical_disagreement ? 'flag' : !rel.sufficient_data && i === 0 ? 'bad' : ''}">${escapeHtml(r)}</li>`));
        if (rel.unused_fields && rel.unused_fields.length) items.push(`<li>${escapeHtml(rel.unused_fields.join(', '))} kaydedildi ancak modelde kullanılmıyor.</li>`);
        if (items.length) {
            relHtml += `<div class="assess-section"><div class="assess-kicker"><span>Neden?</span></div><ul class="reasons">${items.join('')}</ul></div>`;
        }
    } else {
        relHtml = `<div class="assess-section"><p class="small muted">Bu kayıt güvenilirlik hesaplaması eklenmeden önce oluşturuldu. Saati yeniden kaydederseniz güvenilirlik hesaplanır.</p></div>`;
    }

    // --- Özet yorum
    let verdict;
    if (insufficient) {
        verdict = ['none', 'Sonuç yok: hastanın sepsis olmadığı anlamına gelmez. En az 3 temel vital bulgu girip yeniden değerlendirin.'];
    } else if (['high', 'critical'].includes(cls)) {
        verdict = ['danger', rel && rel.clinical_disagreement
            ? 'Yüksek risk (klinik kurallar). Model bu tabloyu yakalayamadı; sepsis protokolüne göre değerlendirin.'
            : 'Yüksek risk. Sepsis protokolüne göre klinik değerlendirme önerilir.'];
    } else if (cls === 'moderate') {
        verdict = ['warn', 'Şüpheli. Yakın takip ve ek laboratuvar (laktat, WBC, kültür) önerilir.'];
    } else if (rel && rel.score < 70) {
        verdict = ['warn', 'Düşük risk görünüyor ancak güvenilirlik sınırlı. Eksik verileri tamamlayıp tekrar değerlendirin; “sepsis yok” olarak yorumlamayın.'];
    } else {
        verdict = ['ok', 'Şu an düşük risk. Model sepsis vakalarının bir kısmını kaçırabilir; klinik şüphe varsa bu sonuç onu geçersiz kılmaz.'];
    }

    body.innerHTML = riskHtml + `<div class="verdict ${verdict[0]}">${verdict[1]}</div>` + relHtml;
}

// ===================== RISK CHART =====================
function renderRiskChart(rows) {
    const canvas = document.getElementById('risk-chart');
    const box = canvas.parentElement;
    if (riskChart) { riskChart.destroy(); riskChart = null; }
    box.querySelector('.chart-empty')?.remove();

    if (!rows.length) {
        box.insertAdjacentHTML('beforeend', '<div class="chart-empty">Kayıt girildikçe risk trendi burada çizilir</div>');
        return;
    }

    const threshold = (modelInfo ? modelInfo.threshold : 0.24) * 100;
    const ok = r => !(r.assessment && !r.assessment.sufficient_data);
    if (!rows.some(ok)) {
        box.insertAdjacentHTML('beforeend', '<div class="chart-empty">Henüz değerlendirilebilir saat yok — kayıtların hepsi “Yetersiz Veri”</div>');
        return;
    }
    const labels = rows.map(r => `S${r.hour}`);
    const score = rows.map(r => ok(r) ? r.prediction * 100 : null);
    const lo = rows.map(r => ok(r) && r.assessment ? r.assessment.mc_dropout.p05 * 100 : null);
    const hi = rows.map(r => ok(r) && r.assessment ? r.assessment.mc_dropout.p95 * 100 : null);
    const primary = cssVar('--primary');
    const grid = cssVar('--border');
    const text = cssVar('--text-3');
    const maxVal = Math.max(threshold * 1.6, ...hi.filter(v => v != null), ...score.filter(v => v != null));

    riskChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'Üst sınır', data: hi, borderWidth: 0, pointRadius: 0, fill: '+1', backgroundColor: primary + '22', spanGaps: true },
                { label: 'Alt sınır', data: lo, borderWidth: 0, pointRadius: 0, fill: false, spanGaps: true },
                {
                    label: 'Risk skoru', data: score, borderColor: primary, borderWidth: 2, tension: 0.25, spanGaps: true,
                    pointRadius: 4, pointBackgroundColor: rows.map(r => ok(r) ? cssVar(`--risk-${riskClass(r.risk_level)}`) || primary : cssVar('--risk-none')),
                    pointBorderWidth: 0,
                },
                { label: 'Eşik', data: rows.map(() => threshold), borderColor: text, borderDash: [5, 4], borderWidth: 1, pointRadius: 0, fill: false },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    filter: item => item.datasetIndex === 2,
                    callbacks: {
                        label: ctx => {
                            const r = rows[ctx.dataIndex];
                            if (!ok(r)) return 'Yetersiz veri';
                            const parts = [`Risk ${ctx.parsed.y.toFixed(1)}%`];
                            if (r.assessment) parts.push(`aralık ${lo[ctx.dataIndex].toFixed(0)}–${hi[ctx.dataIndex].toFixed(0)}%`, `güvenilirlik ${Math.round(r.reliability)}`);
                            return parts.join(' · ');
                        },
                    },
                },
            },
            scales: {
                y: { min: 0, max: Math.min(100, Math.ceil(maxVal / 10) * 10 + 10), grid: { color: grid }, ticks: { color: text, callback: v => v + '%' } },
                x: { grid: { display: false }, ticks: { color: text } },
            },
        },
    });
}

// ===================== FORM =====================
function renderFieldGroups() {
    const container = document.getElementById('field-groups');
    container.innerHTML = FIELD_GROUPS.map(g => `
        <details class="group" ${g.open ? 'open' : ''} data-group="${g.id}">
            <summary>
                <span class="group-title">${g.title}</span>
                ${g.tag ? `<span class="group-tag ${g.tagOff ? 'off' : ''}">${g.tag}</span>` : ''}
                <span class="group-count" id="count-${g.id}">0/${g.fields.length}</span>
            </summary>
            <div class="group-body">
                ${g.note ? `<p class="group-note">${g.note}</p>` : ''}
                <div class="fields">
                    ${g.fields.map(f => `
                        <label class="field">
                            <span class="field-label"><span>${f.label}${f.core ? ' <em>*</em>' : ''}</span><span class="unit">${f.unit}</span></span>
                            <input type="number" id="inp-${f.id}" step="${f.step || 1}" inputmode="decimal"
                                   class="${f.core ? 'core' : ''}" placeholder="${f.range[0]}–${f.range[1]}"
                                   oninput="validateField('${f.id}'); updateCompleteness()">
                            <span class="field-help" id="help-${f.id}">${f.help || ''}</span>
                        </label>`).join('')}
                </div>
            </div>
        </details>`).join('');
}

function validateField(id) {
    const f = FIELD_BY_ID[id];
    const el = document.getElementById(`inp-${id}`);
    const help = document.getElementById(`help-${id}`);
    el.classList.remove('filled', 'out-of-range', 'invalid');
    help.className = 'field-help';
    help.textContent = f.help || '';
    if (el.value === '') return true;

    const v = parseFloat(el.value);
    if (isNaN(v) || v < f.hard[0] || v > f.hard[1]) {
        el.classList.add('invalid');
        help.className = 'field-help err';
        help.textContent = `Kabul edilen aralık ${f.hard[0]}–${f.hard[1]}`;
        return false;
    }
    el.classList.add('filled');
    if (id === 'FiO2' && v > 1) {
        help.textContent = `%${v} → ${(v / 100).toFixed(2)} olarak gönderilecek`;
    } else if (v < f.range[0] || v > f.range[1]) {
        el.classList.add('out-of-range');
        help.className = 'field-help warn';
        help.textContent = `Normal dışı (${f.range[0]}–${f.range[1]})`;
    }
    return true;
}

function formValues() {
    const values = {};
    ALL_FIELDS.forEach(f => {
        const el = document.getElementById(`inp-${f.id}`);
        if (el.value !== '' && !isNaN(parseFloat(el.value))) values[f.id] = parseFloat(el.value);
    });
    return values;
}

function updateCompleteness() {
    const v = formValues();
    const hasMap = v.MAP !== undefined || (v.SBP !== undefined && v.DBP !== undefined);
    const core = CORE_IDS.filter(id => id === 'MAP' ? hasMap : v[id] !== undefined).length;

    FIELD_GROUPS.forEach(g => {
        document.getElementById(`count-${g.id}`).textContent = `${g.fields.filter(f => v[f.id] !== undefined).length}/${g.fields.length}`;
    });

    const keyLabs = FIELD_GROUPS.find(g => g.id === 'keylabs').fields.filter(f => v[f.id] !== undefined).length;
    const pctCore = core / CORE_IDS.length;
    const bar = document.getElementById('completeness-bar');
    bar.style.width = `${Math.round((pctCore * 0.8 + (keyLabs / 6) * 0.2) * 100)}%`;
    bar.style.background = core >= 4 ? 'var(--rel-high)' : core >= MIN_CORE_VITALS ? 'var(--rel-mid)' : 'var(--rel-low)';

    document.getElementById('completeness-text').textContent = `${core} / ${CORE_IDS.length} temel vital · ${keyLabs} / 6 kritik lab`;
    const hint = document.getElementById('completeness-hint');
    if (core < MIN_CORE_VITALS) {
        hint.textContent = `Tahmin için en az ${MIN_CORE_VITALS} temel vital gerekli (${MIN_CORE_VITALS - core} eksik). Eksikler “normal” kabul edilir.`;
        hint.style.color = 'var(--rel-low)';
    } else if (core < CORE_IDS.length) {
        hint.textContent = 'Yeterli. Kalan vitalleri de girmek güvenilirliği artırır.';
        hint.style.color = 'var(--rel-mid)';
    } else {
        hint.textContent = keyLabs ? 'Tüm temel vitaller girildi.' : 'Tüm temel vitaller girildi. Laktat/WBC eklemek güvenilirliği artırır.';
        hint.style.color = 'var(--rel-high)';
    }
}

// ===================== SUBMIT =====================
async function submitHourlyData(event) {
    event.preventDefault();
    if (!currentPatientId) return showToast('Önce bir hasta seçin', 'warning');

    const hourEl = document.getElementById('inp-Hour');
    const hour = parseInt(hourEl.value, 10);
    if (isNaN(hour) || hour < 0) {
        hourEl.focus();
        return showToast('YBÜ kalış saati zorunludur', 'warning');
    }

    const invalid = ALL_FIELDS.filter(f => !validateField(f.id));
    if (invalid.length) {
        const g = FIELD_GROUPS.find(gr => gr.fields.includes(invalid[0]));
        document.querySelector(`details[data-group="${g.id}"]`).open = true;
        document.getElementById(`inp-${invalid[0].id}`).focus();
        return showToast(`${invalid.map(f => f.label).join(', ')}: geçersiz değer`, 'error');
    }

    const values = formValues();
    const modelFields = Object.keys(values).filter(k => !['PCT', 'CRP', 'Presepsin', 'IL6'].includes(k));
    if (!modelFields.length) {
        return showToast('Hiç ölçüm girilmedi. Boş form değerlendirilemez.', 'error');
    }

    if (currentHourly.some(r => r.hour === hour) &&
        !confirm(`Saat ${hour} için kayıt zaten var. Üzerine yazılsın mı?`)) return;

    const btn = document.getElementById('btn-submit');
    btn.disabled = true;
    btn.textContent = 'Değerlendiriliyor…';
    try {
        const r = await fetch(`${API_BASE}/api/patients/${currentPatientId}/hourly-data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hour, vital_signs: values }),
        });
        const d = await r.json();
        if (!d.success) {
            const detail = d.details ? ': ' + d.details.join('; ') : '';
            return showToast((d.error || 'Bir hata oluştu') + detail, 'error');
        }

        const insufficient = d.reliability && !d.reliability.sufficient_data;
        showToast(insufficient ? `Saat ${hour} kaydedildi — yetersiz veri, risk hesaplanmadı`
            : `Saat ${hour} kaydedildi — ${d.risk_level}, güvenilirlik ${Math.round(d.reliability.score)}`,
            insufficient ? 'warning' : 'success');

        await openPatient(currentPatientId);
        renderAssessment(d);
        loadPatients();
    } catch (e) {
        console.error(e);
        showToast('Sunucu bağlantı hatası', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Kaydet ve değerlendir';
    }
}

// ===================== PATIENT CRUD =====================
function showAddPatientModal() {
    document.getElementById('modal-add-patient').hidden = false;
    document.getElementById('new-patient-id').focus();
}

function closeAddPatientModal() {
    document.getElementById('modal-add-patient').hidden = true;
}

async function addPatient(event) {
    event.preventDefault();
    const age = document.getElementById('new-patient-age').value;
    const payload = {
        patient_id: document.getElementById('new-patient-id').value.trim(),
        name: document.getElementById('new-patient-name').value.trim(),
        age: age ? parseInt(age, 10) : null,
        gender: document.getElementById('new-patient-gender').value || null,
    };
    try {
        const r = await fetch(`${API_BASE}/api/patients`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        });
        const d = await r.json();
        if (!d.success) return showToast(d.error || 'Hasta eklenemedi', 'error');
        showToast(`${payload.name} eklendi`, 'success');
        closeAddPatientModal();
        document.getElementById('add-patient-form').reset();
        await loadPatients();
        openPatient(d.patient_id);
    } catch (e) {
        showToast('Sunucu bağlantı hatası', 'error');
    }
}

async function deletePatient(id) {
    const p = allPatients.find(x => x.id === id) || currentPatient;
    if (!confirm(`"${p ? p.name : id}" ve tüm kayıtları kalıcı olarak silinsin mi?`)) return;
    try {
        const r = await fetch(`${API_BASE}/api/patients/${id}`, { method: 'DELETE' });
        const d = await r.json();
        if (!d.success) return showToast(d.error || 'Silinemedi', 'error');
        showToast('Hasta silindi', 'info');
        if (currentPatientId === id) {
            currentPatientId = null;
            document.getElementById('nav-patient-detail').hidden = true;
            switchView('patients');
        }
        loadPatients();
    } catch (e) {
        showToast('Sunucu bağlantı hatası', 'error');
    }
}

function deleteCurrentPatient() { if (currentPatientId) deletePatient(currentPatientId); }

// ===================== MODEL INFO =====================
async function loadModelInfo() {
    const body = document.getElementById('model-info-body');
    try {
        const r = await fetch(`${API_BASE}/api/model-info`);
        modelInfo = await r.json();
        if (!modelInfo.success) throw new Error();
    } catch (e) {
        body.innerHTML = '<div class="card"><div class="empty"><h3>Model bilgisi alınamadı</h3></div></div>';
        return;
    }

    const m = modelInfo.model;
    const st = modelInfo.reliability_stats || {};
    const t = st.test || {};
    const op = t.at_config_threshold;
    const scr = t.at_val_sens80_threshold;
    const metric = (label, value, cls = '') => `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value ${cls}">${value}</div></div>`;
    const grade = (v, good, warn) => v >= good ? 'good' : v >= warn ? 'warn' : 'bad';

    const opRow = (name, x) => x ? `
        <tr><td>${name}</td><td class="num">${x.threshold.toFixed(3)}</td><td class="num">${pct(x.sensitivity)}</td>
        <td class="num">${pct(x.specificity)}</td><td class="num">${pct(x.ppv)}</td><td class="num">${pct(x.npv)}</td>
        <td class="num">${x.tp.toLocaleString('tr-TR')} / ${(x.tp + x.fn).toLocaleString('tr-TR')}</td></tr>` : '';

    body.innerHTML = `
        <div class="card">
            <div class="card-head"><h2>${escapeHtml(m.name)} · ${escapeHtml(m.version)}</h2><span class="legend-note">Test seti: ${st.n_test ? st.n_test.toLocaleString('tr-TR') + ' pencere, prevalans ' + pct(st.prevalence_test) : '—'}</span></div>
            <div class="metrics">
                ${metric('ROC-AUC', (t.roc_auc ?? m.roc_auc).toFixed(3), grade(t.roc_auc ?? m.roc_auc, 0.85, 0.75))}
                ${metric('PR-AUC', (t.pr_auc ?? m.pr_auc).toFixed(3), 'bad')}
                ${op ? metric(`Duyarlılık @${modelInfo.threshold}`, pct(op.sensitivity), grade(op.sensitivity, 0.8, 0.6)) : ''}
                ${op ? metric(`Özgüllük @${modelInfo.threshold}`, pct(op.specificity), grade(op.specificity, 0.9, 0.7)) : ''}
                ${t.ece_raw != null ? metric('ECE (ham → kalibre)', `${t.ece_raw.toFixed(3)} → ${t.ece_calibrated.toFixed(3)}`, 'good') : ''}
                ${metric('Parametre', m.params.toLocaleString('tr-TR'))}
                ${metric('Pencere × özellik', `${modelInfo.window_size} × ${modelInfo.features.length}`)}
                ${st.empty_input ? metric('Boş girdiye skor', pct(st.empty_input.raw), 'warn') : ''}
            </div>
        </div>

        ${op ? `
        <div class="card">
            <div class="card-head"><h2>Karar eşiği seçenekleri (test seti)</h2></div>
            <div class="table-wrap"><table class="table">
                <thead><tr><th>Çalışma noktası</th><th>Eşik</th><th>Duyarlılık</th><th>Özgüllük</th><th>PPV</th><th>NPV</th><th>Yakalanan sepsis</th></tr></thead>
                <tbody>
                    ${opRow('Mevcut (F1 odaklı)', op)}
                    ${opRow('Tarama (val. duyarlılık ≥ %80)', scr)}
                </tbody>
            </table></div>
            <div class="prose">
                <p>Mevcut eşikte model sepsis pencerelerinin yalnızca <strong>${pct(op.sensitivity, 0)}</strong>'ini yakalıyor. “Hasta değil” sonucu bu yüzden tek başına güvence vermez. Erken uyarı amacıyla tarama eşiği kullanılırsa duyarlılık ${scr ? pct(scr.sensitivity, 0) : '—'}'e çıkar, karşılığında yanlış alarm oranı artar (özgüllük ${scr ? pct(scr.specificity, 0) : '—'}).</p>
            </div>
        </div>` : ''}

        ${renderPatientLevel(st)}
        ${renderComparison(st, t)}
        ${renderSanity(st.clinical_sanity)}

        <div class="grid-2">
            <div class="card">
                <div class="card-head"><h2>Güvenilirlik nasıl hesaplanıyor?</h2></div>
                <div class="prose">
                    <p><strong>Minimum veri kapısı.</strong> HR, SpO2, ateş, solunum ve MAP'ten en az 3'ü yoksa risk sınıfı verilmez (“Yetersiz Veri”). Eskiden boş alanlar eğitim medyanıyla doldurulduğu için boş formda bile “düşük risk” çıkıyordu.</p>
                    <p><strong>Veri tamlığı (%45).</strong> Gerçekten girilen değerlerin ağırlıklı oranı: temel vitaller %70, kritik lab %20, diğerleri %10.</p>
                    <p><strong>Zaman kapsamı (%20).</strong> Model 6 saatlik pencere bekler; eksik saatler ilk ölçümün kopyasıdır.</p>
                    <p><strong>Model kararlılığı (%25).</strong> MC-Dropout ile 30 stokastik tahmin. Bunların kaçının aynı tarafta kaldığına bakılır.${st.mc_dropout_check ? ` Testte kararlı tahminlerin doğruluğu ${pct(st.mc_dropout_check.accuracy_high_agreement, 0)}, kararsızlarınki ${pct(st.mc_dropout_check.accuracy_low_agreement, 0)}.` : ''}</p>
                    <p><strong>Dağılım uyumu (%10).</strong> Eğitim dağılımından |z| &gt; 5 uzaklıktaki değerler.</p>
                    <p><strong>Kalibrasyon.</strong> Validasyon setinde izotonik regresyon. Ham skoru gözlenen sepsis oranına çevirir.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-head"><h2>Mimari</h2></div>
                <ol class="arch">${(m.architecture || []).map(a => `<li>${escapeHtml(a)}</li>`).join('')}</ol>
                <div class="prose"><p>Kayıp: ${escapeHtml(m.loss_function)}. Kullanılmayan alanlar: ${escapeHtml(modelInfo.unused_fields.join(', '))}.</p></div>
            </div>
        </div>`;
}

function renderPatientLevel(st) {
    const pl = st.patient_level;
    if (!pl || !pl.length) return '';
    const names = ['Dengeli eşik (F1)', 'Tarama eşiği'];
    return `
        <div class="card">
            <div class="card-head"><h2>Hasta düzeyinde sonuç (test hastaları)</h2>
                <span class="legend-note">${pl[0].n_sepsis_patients} sepsis, ${pl[0].n_non_sepsis_patients.toLocaleString('tr-TR')} sepsis olmayan hasta</span></div>
            <div class="table-wrap"><table class="table">
                <thead><tr><th>Eşik</th><th>Değer</th><th>Sepsis etiketi başlamadan alarm alan sepsis hastası</th><th>En az bir kez alarm alan sepsis olmayan hasta</th></tr></thead>
                <tbody>${pl.map((x, i) => `<tr><td>${names[i] || ''}</td><td class="num">${x.threshold.toFixed(3)}</td>
                    <td class="num">${pct(x.sepsis_patients_alerted, 0)}</td><td class="num">${pct(x.non_sepsis_patients_alerted, 0)}</td></tr>`).join('')}</tbody>
            </table></div>
        </div>`;
}

function renderComparison(st, t) {
    if (!t.roc_auc) return '';
    const lg = st.lightgbm_baseline;
    const rows = [
        ['GRU v3 (aktif)', 'Hasta bazlı', t.roc_auc, t.pr_auc, 'İleri doldurma + ölçüm maskeleri'],
        lg ? ['LightGBM (karşılaştırma)', 'Hasta bazlı, aynı hastalar', lg.roc_auc, lg.pr_auc, 'Aynı özellikler, ağaç tabanlı'] : null,
        ['Optimized v1 (eski)', 'Hasta bazlı', 0.809, 0.089, 'Medyan doldurma; vital bulgulara tepkisiz'],
        ['GRU v23 (ilk sürüm)', 'Rastgele pencere', 0.880, 0.180, 'Aynı hasta hem eğitimde hem testte → iyimser'],
    ].filter(Boolean);
    return `
        <div class="card">
            <div class="card-head"><h2>Model karşılaştırması</h2></div>
            <div class="table-wrap"><table class="table">
                <thead><tr><th>Model</th><th>Test bölmesi</th><th>ROC-AUC</th><th>PR-AUC</th><th>Not</th></tr></thead>
                <tbody>${rows.map(r => `<tr><td><strong>${r[0]}</strong></td><td>${r[1]}</td><td class="num">${r[2].toFixed(3)}</td><td class="num">${r[3].toFixed(3)}</td><td class="muted small">${r[4]}</td></tr>`).join('')}</tbody>
            </table></div>
            <div class="prose"><p>İlk sürümün 0.88'i, veri saat saat rastgele bölündüğü için aynı hastanın bir saati eğitimde, sonraki saati testte bulunarak elde edildi. Yaş + cinsiyet + yatış süresi hastaların %93'ünü tek başına tanımladığından model hastayı “tanıyabiliyordu”. Hasta bazlı bölmede hiçbir test hastası eğitimde görülmez; gerçekçi değer budur.</p></div>
        </div>`;
}

function renderSanity(cs) {
    if (!cs) return '';
    const labels = { HR: 'Nabız 135', Temp: 'Ateş 39.2 °C', Resp: 'Solunum 30/dk', MAP: 'MAP 58', SBP: 'Sistolik KB 80', O2Sat: 'SpO2 %87', Lactate: 'Laktat 5.0', WBC: 'WBC 20' };
    const nv = Object.entries(cs.normal_vs_septic);
    return `
        <div class="card">
            <div class="card-head"><h2>Klinik tutarlılık testi</h2><span class="badge ${cs.passed ? 'very-low' : 'high'}">${cs.passed ? 'Geçti' : 'Kaldı'}</span></div>
            <div class="grid-2" style="padding:0 0 4px">
                <div class="table-wrap"><table class="table">
                    <thead><tr><th>YBÜ saati</th><th>Normal vitaller</th><th>Septik vitaller</th></tr></thead>
                    <tbody>${nv.map(([h, v]) => `<tr><td class="num">${h}</td><td class="num">${pct(v.normal)}</td><td class="num"><strong>${pct(v.septic)}</strong></td></tr>`).join('')}</tbody>
                </table></div>
                <div class="table-wrap"><table class="table">
                    <thead><tr><th>Tek parametre kötüleşince (24. saat)</th><th>Skor değişimi</th></tr></thead>
                    <tbody>${Object.entries(cs.single_feature).map(([k, v]) => `<tr><td>${labels[k] || k}</td><td class="num" style="color:${v.delta >= 0 ? 'var(--risk-very-low)' : 'var(--risk-high)'}">${v.delta >= 0 ? '+' : ''}${(v.delta * 100).toFixed(1)} puan</td></tr>`).join('')}</tbody>
                </table></div>
            </div>
            <div class="prose"><p>Eski model bu testte septik hastaya normal hastayla neredeyse aynı skoru veriyordu (%7 civarı). Model laboratuvar değerinin kendisini değil “ölçülmüş olmasını” öğrenmişti.</p></div>
        </div>`;
}

// ===================== TOAST =====================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.transition = 'opacity 250ms';
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 250);
    }, 4500);
}
