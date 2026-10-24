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
let currentUser = null;
let patientStatus = 'active';
let archivedPatients = [];
let alarms = [];
let alarmStatus = 'open';
let auditOffset = 0;

// ===================== FORM ŞEMASI =====================
// range: klinik normal aralık (yer tutucu ve uyarı için), hard: sunucunun kabul ettiği sınırlar
// Hızlı giriş: 6 vital + FiO2 + 4 kritik lab. Test setinde (8.066 görülmemiş hasta) bu 11 değerle
// ROC-AUC 0.805, 34 değerin tamamıyla 0.806, FiO2 olmadan 10 değerle 0.785. Kalanlar isteğe bağlı.
const FIELD_GROUPS = [
    {
        id: 'vitals', title: 'Ana değerler — vital bulgular ve oksijen', open: true, tag: 'En az 3 zorunlu',
        note: 'Tansiyondan ortalama arter basıncı (MAP) otomatik hesaplanır.',
        fields: [
            { id: 'HR', label: 'Nabız', unit: 'atım/dk', range: [60, 100], hard: [40, 200], core: true },
            { id: 'O2Sat', label: 'SpO2', unit: '%', range: [95, 100], hard: [70, 100], core: true },
            { id: 'Temp', label: 'Ateş', unit: '°C', range: [36, 37.5], hard: [35, 42], step: 0.1, core: true },
            { id: 'Resp', label: 'Solunum', unit: '/dk', range: [12, 20], hard: [8, 50], core: true },
            { id: 'SBP', label: 'Sistolik KB', unit: 'mmHg', range: [100, 140], hard: [60, 250], core: true },
            { id: 'DBP', label: 'Diyastolik KB', unit: 'mmHg', range: [60, 90], hard: [30, 150], core: true },
            { id: 'FiO2', label: 'Oksijen desteği (FiO2)', unit: '%', range: [21, 50], hard: [0.21, 100], step: 1,
              help: 'Yalnızca oksijen/ventilatör desteği varsa girin' },
        ],
    },
    {
        id: 'keylabs', title: 'Ana değerler — kritik laboratuvar', open: true, tag: 'Varsa girin',
        note: 'Son 24 saatte ölçüldüyse girin. Sonraki saatlerde değer otomatik olarak geçerli kabul edilir.',
        fields: [
            { id: 'Lactate', label: 'Laktat', unit: 'mmol/L', range: [0.5, 2], hard: [0.5, 20], step: 0.1 },
            { id: 'WBC', label: 'WBC', unit: '10³/µL', range: [4, 12], hard: [1, 50], step: 0.1 },
            { id: 'Creatinine', label: 'Kreatinin', unit: 'mg/dL', range: [0.6, 1.2], hard: [0.3, 15], step: 0.01 },
            { id: 'Platelets', label: 'Trombosit', unit: '10³/µL', range: [150, 400], hard: [20, 800] },
        ],
    },
    {
        id: 'x-gas', extra: true, open: true, title: 'Kan gazı ve solunum',
        fields: [
            { id: 'pH', label: 'pH', unit: '', range: [7.35, 7.45], hard: [6.8, 7.8], step: 0.01 },
            { id: 'PaCO2', label: 'PaCO2', unit: 'mmHg', range: [35, 45], hard: [15, 100] },
            { id: 'HCO3', label: 'HCO3', unit: 'mmol/L', range: [22, 26], hard: [10, 45], step: 0.1 },
            { id: 'BaseExcess', label: 'Baz açığı', unit: 'mmol/L', range: [-2, 2], hard: [-20, 20], step: 0.1 },
            { id: 'SaO2', label: 'SaO2', unit: '%', range: [95, 100], hard: [50, 100] },
            { id: 'EtCO2', label: 'EtCO2', unit: 'mmHg', range: [35, 45], hard: [10, 80] },
            { id: 'MAP', label: 'MAP (elle)', unit: 'mmHg', range: [70, 100], hard: [40, 180], help: 'Boşsa tansiyondan hesaplanır' },
        ],
    },
    {
        id: 'x-chem', extra: true, open: true, title: 'Biyokimya',
        fields: [
            { id: 'BUN', label: 'BUN', unit: 'mg/dL', range: [7, 20], hard: [3, 150] },
            { id: 'Glucose', label: 'Glukoz', unit: 'mg/dL', range: [70, 140], hard: [30, 600] },
            { id: 'Bilirubin_total', label: 'Total bilirubin', unit: 'mg/dL', range: [0.1, 1.2], hard: [0.1, 30], step: 0.1 },
            { id: 'Potassium', label: 'Potasyum', unit: 'mmol/L', range: [3.5, 5], hard: [2, 8], step: 0.1 },
            { id: 'Chloride', label: 'Klor', unit: 'mmol/L', range: [96, 106], hard: [70, 130] },
            { id: 'Calcium', label: 'Kalsiyum', unit: 'mg/dL', range: [8.5, 10.5], hard: [5, 15], step: 0.1 },
            { id: 'Magnesium', label: 'Magnezyum', unit: 'mg/dL', range: [1.7, 2.4], hard: [0.5, 5], step: 0.1 },
            { id: 'Phosphate', label: 'Fosfat', unit: 'mg/dL', range: [2.5, 4.5], hard: [0.5, 15], step: 0.1 },
            { id: 'AST', label: 'AST', unit: 'U/L', range: [10, 40], hard: [5, 5000] },
            { id: 'Alkalinephos', label: 'ALP', unit: 'U/L', range: [40, 130], hard: [10, 2000] },
            { id: 'Bilirubin_direct', label: 'Direkt bilirubin', unit: 'mg/dL', range: [0, 0.3], hard: [0, 15], step: 0.1 },
        ],
    },
    {
        id: 'x-heme', extra: true, open: true, title: 'Hematoloji ve pıhtılaşma',
        fields: [
            { id: 'Hgb', label: 'Hemoglobin', unit: 'g/dL', range: [12, 17], hard: [5, 20], step: 0.1 },
            { id: 'Hct', label: 'Hematokrit', unit: '%', range: [36, 50], hard: [15, 65], step: 0.1 },
            { id: 'PTT', label: 'PTT', unit: 'sn', range: [25, 35], hard: [15, 150], step: 0.1 },
            { id: 'Fibrinogen', label: 'Fibrinojen', unit: 'mg/dL', range: [200, 400], hard: [30, 1200] },
            { id: 'TroponinI', label: 'Troponin I', unit: 'ng/mL', range: [0, 0.04], hard: [0, 100], step: 0.01 },
        ],
    },
    {
        id: 'x-bio', extra: true, open: true, title: 'Biyobelirteçler', tag: 'Modelde kullanılmaz', tagOff: true,
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
// Güvenilirlik kapısındaki 5 temel vital (sunucudaki reliability.CORE_VITALS ile aynı); MAP tansiyondan türetilir
const CORE_IDS = ['HR', 'O2Sat', 'Temp', 'Resp', 'MAP'];
const KEY_LAB_IDS = FIELD_GROUPS.find(g => g.id === 'keylabs').fields.map(f => f.id);
const QUICK_COUNT = FIELD_GROUPS.filter(g => !g.extra).reduce((n, g) => n + g.fields.length, 0);

const COMPONENT_LABELS = {
    completeness: 'Veri tamlığı',
    temporal: 'Zaman kapsamı',
    certainty: 'Model kararlılığı',
    in_distribution: 'Dağılım uyumu',
};

// ===================== API =====================
async function api(path, options = {}) {
    const opts = { credentials: 'same-origin', ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } };
    if (opts.body && typeof opts.body !== 'string') opts.body = JSON.stringify(opts.body);
    const r = await fetch(API_BASE + path, opts);
    let d = {};
    try { d = await r.json(); } catch (e) { }
    if (r.status === 401 && path !== '/api/auth/me') {
        window.location.href = '/login?expired=1';
        throw new Error('Oturum sona erdi');
    }
    if (!r.ok || d.success === false) {
        const err = new Error(d.error || `İstek başarısız (${r.status})`);
        err.details = d.details;
        err.status = r.status;
        throw err;
    }
    return d;
}

function can(permission) { return !!currentUser && currentUser.permissions.includes(permission); }

function errorText(e) { return e.message + (e.details ? ': ' + e.details.join('; ') : ''); }

// ===================== INIT =====================
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const me = await api('/api/auth/me');
        currentUser = me.user;
        document.getElementById('public-demo-banner').hidden = !me.public_demo;
    } catch (e) {
        window.location.href = '/login';
        return;
    }
    renderUserCard();
    applyPermissions(document);
    renderFieldGroups();
    startClock();
    checkHealth();
    loadPatients();
    loadAlarms();
    loadModelInfo();
    setInterval(() => { loadPatients(); loadAlarms(); }, 60000);

    document.querySelectorAll('.modal').forEach(m => m.addEventListener('click', e => {
        if (e.target === m) m.hidden = true;
    }));
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') document.querySelectorAll('.modal').forEach(m => { m.hidden = true; });
    });
});

function renderUserCard() {
    document.getElementById('user-name').textContent = currentUser.full_name;
    const role = document.getElementById('user-role');
    role.textContent = currentUser.role_label;
    role.className = `role-badge role-${currentUser.role}`;
    document.getElementById('user-avatar').textContent = initials(currentUser.full_name);
}

function applyPermissions(root) {
    root.querySelectorAll('[data-perm]').forEach(el => { el.hidden = !can(el.dataset.perm); });
}

function initials(name) {
    return (name || '?').replace(/^(Dr\.|Hemşire)\s+/i, '').split(/\s+/).slice(0, 2)
        .map(w => w.charAt(0).toLocaleUpperCase('tr-TR')).join('');
}

async function logout() {
    try { await api('/api/auth/logout', { method: 'POST' }); } catch (e) { }
    window.location.href = '/login';
}

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
        d.preprocessing_loaded = d.preprocessing_loaded ?? d.model_loaded;
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
    'patients': ['Hastalar', 'Yatan hastalar ve arşiv'],
    'patient-detail': ['Hasta', 'Saatlik veri girişi, risk ve güvenilirlik'],
    'alerts': ['Alarmlar', 'Yüksek risk uyarıları ve doktor onayı'],
    'model-info': ['Model & Güvenilirlik', 'Aktif modelin ölçülmüş performansı ve kalibrasyonu'],
    'users': ['Kullanıcılar', 'Hesaplar, roller ve erişim'],
    'audit': ['Denetim kayıtları', 'Sistemde kim, ne zaman, ne yaptı'],
    'database': ['Veritabanı', 'Tablo yapısı, ilişkiler ve kayıt sayıları'],
};

function switchView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${name}`));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === name));
    const [title, sub] = VIEW_TITLES[name] || ['', ''];
    document.getElementById('page-title').textContent = title;
    document.getElementById('page-subtitle').textContent = sub;
    document.getElementById('sidebar').classList.remove('open');
    window.scrollTo({ top: 0 });
    if (name === 'users') loadUsers();
    if (name === 'audit') loadAudit();
    if (name === 'database') loadDatabase();
    if (name === 'alerts') loadAlarms();
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

function fmtNum(v) { return typeof v === 'number' ? String(Math.round(v * 100) / 100) : v; }

function pct(v, digits = 1) { return v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`; }

function avatarHtml(p) {
    return `<div class="avatar">${escapeHtml((p.name || '?').trim().charAt(0).toLocaleUpperCase('tr-TR'))}</div>`;
}

function timeAgo(ts) {
    if (!ts) return '—';
    const d = parseTs(ts);
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
        allPatients = (await api('/api/patients?status=active')).patients || [];
        if (patientStatus === 'discharged') archivedPatients = (await api('/api/patients?status=discharged')).patients || [];
        renderDashboard();
        renderAllPatients();
        renderFollowUp();
    } catch (e) {
        console.error(e);
        showToast('Hasta verileri yüklenemedi: ' + e.message, 'error');
    }
}

function setPatientStatus(status) {
    patientStatus = status;
    document.querySelectorAll('#view-patients .tab').forEach(t => t.classList.toggle('active', t.dataset.status === status));
    loadPatients();
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
        tbody.innerHTML = `<tr><td colspan="5"><div class="empty"><h3>Yatan hasta yok</h3><p>${can('patients.create') ? 'Sağ üstteki “Yeni Hasta” ile başlayın.' : 'Hasta kabulü hemşire veya doktor tarafından yapılır.'}</p></div></td></tr>`;
        return;
    }
    tbody.innerHTML = sortedByRisk(allPatients).slice(0, 10).map(p => `
        <tr class="clickable" onclick="openPatient(${p.id})">
            <td><div class="person">${avatarHtml(p)}<div><strong>${escapeHtml(p.name)}${p.open_alarms ? ' <span class="alarm-dot" title="Açık alarm"></span>' : ''}</strong><span>${escapeHtml(p.patient_id)} · ${p.age || '—'} yaş</span></div></div></td>
            <td>${riskBadge(p.latest_risk)}</td>
            <td class="num">${riskClass(p.latest_risk) === 'none' ? '—' : pct(p.latest_prediction)}</td>
            <td>${relBadge(p.latest_reliability, riskClass(p.latest_risk) === 'none')}</td>
            <td class="muted small">${p.latest_hour != null ? `Saat ${p.latest_hour} · ${timeAgo(p.latest_timestamp)}` : 'Ölçüm yok'}</td>
        </tr>`).join('');
}

function renderAllPatients() {
    const source = patientStatus === 'active' ? allPatients : archivedPatients;
    const q = (document.getElementById('patient-search').value || '').toLocaleLowerCase('tr-TR');
    const list = source.filter(p => !q ||
        (p.name || '').toLocaleLowerCase('tr-TR').includes(q) || (p.patient_id || '').toLocaleLowerCase('tr-TR').includes(q));
    const tbody = document.getElementById('all-patients-body');
    if (!list.length) {
        const empty = source.length ? 'Eşleşen hasta yok' : patientStatus === 'active' ? 'Yatan hasta yok' : 'Arşivde hasta yok';
        tbody.innerHTML = `<tr><td colspan="8"><div class="empty"><h3>${empty}</h3></div></td></tr>`;
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
            <td>${p.status === 'discharged' ? `<span class="badge none">Taburcu · ${formatDate(p.discharged_at)}</span>${p.outcome_sepsis ? `<br><span class="small ${p.outcome_sepsis === 'yes' ? 'text-danger' : 'muted'}">${OUTCOME_LABELS[p.outcome_sepsis]}</span>` : '<br><span class="small muted">Sonuç girilmemiş</span>'}` : p.open_alarms ? '<span class="badge high">Açık alarm</span>' : '<span class="muted small">Yatıyor</span>'}</td>
        </tr>`).join('');
}

function renderFollowUp() {
    const insufficient = allPatients.filter(p => (p.latest_risk || '').includes('Yetersiz'));
    const lowRel = allPatients.filter(p => riskClass(p.latest_risk) !== 'none' && p.latest_reliability != null && p.latest_reliability < 45);
    const noData = allPatients.filter(p => p.latest_hour == null);
    const row = (p, extra) => `
        <div class="alert-row" onclick="openPatient(${p.id})">
            <div class="person">${avatarHtml(p)}<div><strong>${escapeHtml(p.name)}</strong><span>${escapeHtml(p.patient_id)} · Saat ${p.latest_hour ?? '—'}</span></div></div>
            ${extra}
        </div>`;
    const html = insufficient.map(p => row(p, '<span class="badge none">Yetersiz veri — tekrar ölçüm girin</span>')).join('')
        + lowRel.map(p => row(p, `${riskBadge(p.latest_risk)}<span class="rel low">Düşük güvenilirlik</span>`)).join('')
        + noData.map(p => row(p, '<span class="badge none">Henüz ölçüm yok</span>')).join('');
    document.getElementById('followup-body').innerHTML = html ||
        '<div class="empty"><p>Takip gerektiren hasta yok.</p></div>';
}

// ===================== ALARMS =====================
async function loadAlarms() {
    try {
        const open = (await api('/api/alarms?status=open')).alarms || [];
        const badge = document.getElementById('alert-count');
        badge.hidden = open.length === 0;
        badge.textContent = open.length;
        const banner = document.getElementById('critical-alert');
        banner.hidden = open.length === 0;
        document.getElementById('critical-alert-text').textContent =
            `${open.length} açık alarm${can('alarms.acknowledge') ? ' doktor onayı bekliyor.' : ' — doktor onayı bekleniyor.'}`;
        alarms = alarmStatus === 'open' ? open : ((await api(`/api/alarms?status=${alarmStatus}`)).alarms || []);
        renderAlarms();
    } catch (e) {
        console.error(e);
    }
}

function setAlarmStatus(status) {
    alarmStatus = status;
    document.querySelectorAll('#view-alerts .tab').forEach(t => t.classList.toggle('active', t.dataset.alarm === status));
    loadAlarms();
}

function renderAlarms() {
    document.getElementById('alarm-role-note').textContent = can('alarms.acknowledge')
        ? 'Onaylamak için kısa bir klinik not yazın' : 'Alarmları yalnızca doktorlar onaylayabilir';
    const body = document.getElementById('alerts-body');
    if (!alarms.length) {
        body.innerHTML = `<div class="empty"><h3>${alarmStatus === 'open' ? 'Açık alarm yok' : 'Onaylanmış alarm yok'}</h3><p>Yüksek veya çok yüksek risk hesaplandığında burada alarm açılır.</p></div>`;
        return;
    }
    body.innerHTML = alarms.map(a => `
        <div class="alarm-row">
            <div class="person clickable" onclick="openPatient(${a.patient_id})">
                <div class="avatar">${escapeHtml(initials(a.patient_name))}</div>
                <div><strong>${escapeHtml(a.patient_name)}</strong><span>${escapeHtml(a.patient_code)} · Saat ${a.hour ?? '—'} · ${formatDateTime(a.created_at)}</span></div>
            </div>
            <div class="alarm-risk">${riskBadge(a.risk_level)}<span class="num">${pct(a.score)}</span></div>
            ${a.status === 'open'
                ? (can('alarms.acknowledge')
                    ? `<form class="ack-form" onsubmit="acknowledgeAlarm(event, ${a.id})">
                           <input type="text" placeholder="Klinik not (ör. laktat ve kültür istendi)" required minlength="3">
                           <button class="btn btn-primary btn-sm">Onayla</button>
                       </form>`
                    : '<span class="muted small">Doktor onayı bekleniyor</span>')
                : `<div class="ack-done"><strong>${escapeHtml(a.acknowledged_by_name || '—')}</strong> · ${formatDateTime(a.acknowledged_at)}<br><span class="muted">${escapeHtml(a.note || '')}</span></div>`}
        </div>`).join('');
}

async function acknowledgeAlarm(event, alarmId) {
    event.preventDefault();
    const note = event.target.querySelector('input').value.trim();
    try {
        await api(`/api/alarms/${alarmId}/acknowledge`, { method: 'POST', body: { note } });
        showToast('Alarm onaylandı', 'success');
        loadAlarms();
        loadPatients();
        if (currentPatientId) openPatient(currentPatientId, { keepScroll: true });
    } catch (e) {
        showToast(errorText(e), 'error');
    }
}

// ===================== PATIENT DETAIL =====================
async function openPatient(id, opts = {}) {
    currentPatientId = id;
    document.getElementById('nav-patient-detail').hidden = false;
    if (!opts.keepScroll) switchView('patient-detail');

    try {
        const d = await api(`/api/patients/${id}`);
        currentPatient = d.patient;
        currentHourly = d.hourly_data || [];
        const discharged = currentPatient.status === 'discharged';

        document.getElementById('nav-patient-name').textContent = currentPatient.name;
        document.getElementById('detail-name').textContent = currentPatient.name;
        document.getElementById('detail-meta').innerHTML =
            `ID ${escapeHtml(currentPatient.patient_id)} · ${currentPatient.age ? currentPatient.age + ' yaş' : 'yaş girilmemiş'} · ${genderText(currentPatient.gender)} · ${currentHourly.length} saatlik kayıt`
            + (currentPatient.created_by_name ? ` · Kabul: ${escapeHtml(currentPatient.created_by_name)}` : '')
            + (discharged ? ` · <span class="badge none">Taburcu ${formatDateTime(currentPatient.discharged_at)}${currentPatient.discharged_by_name ? ' — ' + escapeHtml(currentPatient.discharged_by_name) : ''}</span>` : '')
            + (currentPatient.outcome ? ` · <span class="badge ${currentPatient.outcome.sepsis === 'yes' ? 'high' : 'none'}">${OUTCOME_LABELS[currentPatient.outcome.sepsis]}${onsetText(currentPatient.outcome)}</span>` : '');
        document.getElementById('detail-avatar').textContent = initials(currentPatient.name);

        renderPatientActions(discharged);
        renderPatientAlarm(d.open_alarm);

        const canEnter = can('observations.create') && !discharged;
        document.getElementById('entry-card').hidden = !canEnter;
        document.getElementById('entry-locked').hidden = canEnter;
        if (!canEnter) {
            document.getElementById('entry-locked-title').textContent = discharged ? 'Hasta taburcu edildi' : 'Veri girişi bu rol için kapalı';
            document.getElementById('entry-locked-text').textContent = discharged
                ? 'Arşivdeki hastaya veri girilemez. Gerekirse doktor hastayı yeniden kabul edebilir.'
                : 'Saatlik ölçümleri hemşire veya doktor girer. Yönetici kayıtları görüntüleyebilir.';
        } else {
            resetEntryForm();
        }
        renderHourlyHistory(currentHourly);
        renderRiskChart(currentHourly);

        const last = currentHourly[currentHourly.length - 1];
        if (last) {
            renderAssessment({ prediction: last.prediction, risk_level: last.risk_level, hour: last.hour, reliability: last.assessment });
        } else {
            document.getElementById('assess-empty').hidden = false;
            document.getElementById('assess-body').hidden = true;
        }
        if (can('history.view')) loadPatientHistory(id);
    } catch (e) {
        console.error(e);
        showToast('Hasta bilgileri yüklenemedi: ' + e.message, 'error');
    }
}

function renderPatientActions(discharged) {
    const parts = ['<button class="btn btn-ghost" onclick="switchView(\'patients\')">← Liste</button>'];
    if (can('patients.discharge')) {
        parts.push(discharged
            ? '<button class="btn btn-ghost" onclick="readmitPatient()">Yeniden kabul et</button>'
            : '<button class="btn btn-ghost" onclick="dischargePatient()">Taburcu et</button>');
    }
    if (discharged && can('outcomes.label')) parts.push('<button class="btn btn-ghost" onclick="editOutcome()">Sonucu düzenle</button>');
    if (can('patients.delete')) parts.push('<button class="btn btn-danger-ghost" onclick="deleteCurrentPatient()">Kalıcı sil</button>');
    document.getElementById('patient-actions').innerHTML = parts.join('');
}

function renderPatientAlarm(alarm) {
    const box = document.getElementById('patient-alarm');
    if (!alarm) { box.innerHTML = ''; return; }
    box.innerHTML = `
        <div class="banner banner-danger patient-alarm">
            <div><strong>Açık alarm:</strong> ${escapeHtml(alarm.risk_level)} · skor ${pct(alarm.score)} · ${formatDateTime(alarm.created_at)}</div>
            ${can('alarms.acknowledge')
                ? `<form class="ack-form" onsubmit="acknowledgeAlarm(event, ${alarm.id})">
                       <input type="text" placeholder="Klinik not" required minlength="3">
                       <button class="btn btn-primary btn-sm">Onayla</button></form>`
                : '<span class="small">Doktor onayı bekleniyor</span>'}
        </div>`;
}

const ACTION_LABELS = {
    'auth.login': 'Giriş yaptı', 'auth.logout': 'Çıkış yaptı', 'auth.login_failed': 'Başarısız giriş denemesi',
    'auth.locked': 'Hesap kilitlendi', 'auth.forbidden': 'Yetkisiz işlem denemesi', 'auth.password_change': 'Şifre değiştirdi',
    'patient.create': 'Hasta kabul etti', 'patient.discharge': 'Taburcu etti', 'patient.readmit': 'Yeniden kabul etti',
    'patient.delete': 'Hastayı kalıcı sildi', 'observation.create': 'Saatlik veri girdi', 'observation.update': 'Saatlik veriyi güncelledi',
    'patient.outcome': 'Sonucu güncelledi', 'dataset.export': 'Veri setini dışa aktardı',
    'alarm.raised': 'Alarm açıldı', 'alarm.acknowledge': 'Alarmı onayladı', 'user.create': 'Kullanıcı ekledi',
    'user.update': 'Kullanıcıyı güncelledi', 'users.demo_created': 'Demo hesapları oluşturuldu', 'db.migrate': 'Veritabanı taşındı',
};

function describeEvent(action, d) {
    if (!d) return '';
    if (action === 'observation.create' || action === 'observation.update') {
        let text = `Saat ${d.hour}: ${Object.keys(d.values || {}).length} değer → ${d.risk_level} (%${((d.score || 0) * 100).toFixed(1)})`;
        if (d.old_values) {
            const changed = Object.keys({ ...d.old_values, ...d.values })
                .filter(k => d.old_values[k] !== d.values[k])
                .map(k => `${k} ${d.old_values[k] ?? '—'}→${d.values[k] ?? '—'}`);
            if (changed.length) text += ` · değişen: ${changed.join(', ')}`;
        }
        return text;
    }
    if (action === 'alarm.acknowledge') return `Not: ${d.note}`;
    if (action === 'alarm.raised') return `${d.risk_level} · saat ${d.hour}`;
    if (action === 'patient.create') return `${d.patient_id} · ${d.name}`;
    if (action === 'patient.discharge' || action === 'patient.outcome') {
        const parts = [];
        if (d.sepsis) parts.push(OUTCOME_LABELS[d.sepsis] + onsetText(d));
        if (d.old) parts.push(`önceki: ${OUTCOME_LABELS[d.old.sepsis]}`);
        if (d.note) parts.push(`Not: ${d.note}`);
        if (d.closed_alarms) parts.push(`${d.closed_alarms} alarm kapatıldı`);
        return parts.join(' · ');
    }
    if (action === 'dataset.export') return `${d.rows} satır${d.patients != null ? `, ${d.patients} hasta` : ''}`;
    if (action === 'patient.delete') return `${d.patient_id} · ${d.name} · ${d.deleted_hours} saatlik kayıt`;
    if (action === 'auth.login_failed') return `Kullanıcı: ${d.username} · ${d.reason}`;
    if (action === 'auth.forbidden') return `${d.permission} · ${d.path}`;
    if (action === 'user.create') return `${d.username} (${d.role})`;
    if (action === 'user.update') return Object.entries(d).filter(([k]) => k !== 'username')
        .map(([k, v]) => Array.isArray(v) ? `${k}: ${v[0]}→${v[1]}` : k).join(', ') + ` · ${d.username}`;
    return '';
}

async function loadPatientHistory(id) {
    const box = document.getElementById('patient-history');
    try {
        const history = (await api(`/api/patients/${id}/history`)).history || [];
        if (!history.length) { box.innerHTML = '<div class="empty"><p>Bu hasta için kayıt yok.</p></div>'; return; }
        box.innerHTML = history.map(h => `
            <div class="timeline-item ${h.success ? '' : 'failed'}">
                <span class="timeline-time">${formatDateTime(h.timestamp)}</span>
                <span class="timeline-who">${escapeHtml(h.username || 'Sistem')}${h.role && h.role !== 'system' ? ` <span class="role-badge role-${h.role}">${roleLabel(h.role)}</span>` : ''}</span>
                <span class="timeline-what"><strong>${ACTION_LABELS[h.action] || h.action}</strong> ${escapeHtml(describeEvent(h.action, h.details))}</span>
            </div>`).join('');
    } catch (e) {
        box.innerHTML = `<div class="empty"><p>${escapeHtml(e.message)}</p></div>`;
    }
}

function roleLabel(role) { return { admin: 'Yönetici', doctor: 'Doktor', nurse: 'Hemşire' }[role] || role; }

function parseTs(ts) { return ts ? new Date(ts.replace(' ', 'T')) : null; }
function formatDateTime(ts) {
    const d = parseTs(ts);
    return d && !isNaN(d) ? d.toLocaleString('tr-TR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—';
}
function formatDate(ts) {
    const d = parseTs(ts);
    return d && !isNaN(d) ? d.toLocaleDateString('tr-TR') : '—';
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
        tbody.innerHTML = `<tr><td colspan="7"><div class="empty"><h3>Henüz kayıt yok</h3><p>Yukarıdaki formla ilk saatlik veriyi girin.</p></div></td></tr>`;
        return;
    }
    tbody.innerHTML = [...rows].reverse().map(e => {
        const v = e.vital_signs || {};
        const chips = [['HR', 'HR'], ['O2Sat', 'SpO2'], ['Temp', 'T'], ['Resp', 'SS'], ['MAP', 'MAP'], ['Lactate', 'Lak'], ['WBC', 'WBC']]
            .filter(([k]) => v[k] !== undefined && v[k] !== null)
            .map(([k, l]) => `<span class="chip"><b>${l}</b>${escapeHtml(fmtNum(v[k]))}</span>`).join('');
        const insufficient = e.assessment ? !e.assessment.sufficient_data : false;
        return `
            <tr>
                <td class="num">${e.hour}</td>
                <td><div class="vitals">${chips || '<span class="muted">—</span>'}</div></td>
                <td class="num">${insufficient ? '—' : pct(e.prediction)}</td>
                <td>${riskBadge(e.risk_level)}</td>
                <td>${relBadge(e.reliability, insufficient)}</td>
                <td class="small">${e.entered_by ? escapeHtml(e.entered_by) + `<br><span class="muted">${formatDateTime(e.timestamp)}</span>` : '<span class="muted">—</span>'}</td>
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
                    ${rel && rel.components_model ? `Model dalları: LightGBM <b>${pct(rel.components_model.lightgbm)}</b> · sinir ağı <b>${pct(rel.components_model.neural_net)}</b><br>` : ''}
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

        const ex = rel.explanation;
        if (ex && (ex.factors_up.length || ex.factors_down.length)) {
            const maxPts = Math.max(1, ...[...ex.factors_up, ...ex.factors_down].map(f => Math.abs(f.points)));
            const factorRow = f => `
                <div class="factor ${f.points >= 0 ? 'up' : 'down'}">
                    <div class="factor-text"><strong>${escapeHtml(f.label)}</strong><span>${escapeHtml(f.detail)}</span></div>
                    <div class="factor-bar"><i style="width:${Math.max(4, Math.abs(f.points) / maxPts * 100)}%"></i></div>
                    <b class="num">${f.points >= 0 ? '+' : ''}${f.points.toFixed(1)}</b>
                </div>`;
            relHtml += `
                <div class="assess-section">
                    <div class="assess-kicker"><span>Skoru etkileyen faktörler</span><span title="${escapeHtml(ex.method)}">yaklaşık puan</span></div>
                    ${ex.factors_up.length ? `<div class="factor-group-title">Riski artıran</div>${ex.factors_up.map(factorRow).join('')}` : ''}
                    ${ex.factors_down.length ? `<div class="factor-group-title">Riski azaltan</div>${ex.factors_down.map(factorRow).join('')}` : ''}
                    <p class="factor-note">LightGBM dalının karar ağaçlarından hesaplanır; büyüklükler yaklaşıktır.</p>
                </div>`;
        }

        const items = [];
        (rel.red_flags || []).forEach(f => items.push(`<li class="flag">Klinik bayrak: ${escapeHtml(FIELD_BY_ID[f.feature]?.label || f.feature)} = ${escapeHtml(f.value)}${f.measured_hour != null && f.measured_hour !== res.hour ? ` (saat ${f.measured_hour} ölçümü)` : ''}</li>`));
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
function fieldHtml(f) {
    return `
        <label class="field">
            <span class="field-label"><span>${f.label}${f.core ? ' <em>*</em>' : ''}</span><span class="unit">${f.unit}</span></span>
            <input type="number" id="inp-${f.id}" step="${f.step || 1}" inputmode="decimal"
                   class="${f.core ? 'core' : ''}" placeholder="${f.range[0]}–${f.range[1]}"
                   oninput="validateField('${f.id}'); updateCompleteness()">
            <span class="field-help" id="help-${f.id}">${f.help || ''}</span>
        </label>`;
}

function groupHtml(g) {
    return `
        <details class="group" ${g.open ? 'open' : ''} data-group="${g.id}">
            <summary>
                <span class="group-title">${g.title}</span>
                ${g.tag ? `<span class="group-tag ${g.tagOff ? 'off' : ''}">${g.tag}</span>` : ''}
                <span class="group-count" id="count-${g.id}">0/${g.fields.length}</span>
            </summary>
            <div class="group-body">
                ${g.note ? `<p class="group-note">${g.note}</p>` : ''}
                <div class="fields">${g.fields.map(fieldHtml).join('')}</div>
            </div>
        </details>`;
}

function renderFieldGroups() {
    const quick = FIELD_GROUPS.filter(g => !g.extra);
    const extra = FIELD_GROUPS.filter(g => g.extra);
    const extraCount = extra.reduce((n, g) => n + g.fields.length, 0);
    document.getElementById('field-groups').innerHTML = quick.map(groupHtml).join('') + `
        <details class="group extras" data-group="extras" open>
            <summary>
                <span class="group-title">Ek değerler</span>
                <span class="group-tag off">Varsa girin</span>
                <span class="group-count" id="count-extras">0/${extraCount}</span>
            </summary>
            <div class="group-body">
                <p class="group-note">Kan gazı, biyokimya ve hematoloji sonuçları varsa girin. Model hepsini kullanır;
                yukarıdaki ana değerler modelin analizi için en önemli olanlardır.</p>
                ${extra.map(groupHtml).join('')}
            </div>
        </details>`;
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
        help.textContent = `%${v} → ${(v / 100).toFixed(2)} oranı olarak gönderilecek`;
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
    const extraFields = FIELD_GROUPS.filter(g => g.extra).flatMap(g => g.fields);
    document.getElementById('count-extras').textContent = `${extraFields.filter(f => v[f.id] !== undefined).length}/${extraFields.length}`;

    const keyLabs = KEY_LAB_IDS.filter(id => v[id] !== undefined).length;
    const pctCore = core / CORE_IDS.length;
    const bar = document.getElementById('completeness-bar');
    bar.style.width = `${Math.round((pctCore * 0.8 + (keyLabs / KEY_LAB_IDS.length) * 0.2) * 100)}%`;
    bar.style.background = core >= 4 ? 'var(--rel-high)' : core >= MIN_CORE_VITALS ? 'var(--rel-mid)' : 'var(--rel-low)';

    document.getElementById('completeness-text').textContent = `${core} / ${CORE_IDS.length} temel vital · ${keyLabs} / ${KEY_LAB_IDS.length} kritik lab`;
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
        if (g.extra) document.querySelector('details[data-group="extras"]').open = true;
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
        const d = await api(`/api/patients/${currentPatientId}/hourly-data`, {
            method: 'POST', body: { hour, vital_signs: values },
        });

        const insufficient = d.reliability && !d.reliability.sufficient_data;
        showToast(insufficient ? `Saat ${hour} kaydedildi — yetersiz veri, risk hesaplanmadı`
            : `Saat ${hour} kaydedildi — ${d.risk_level}, güvenilirlik ${Math.round(d.reliability.score)}`,
            insufficient ? 'warning' : 'success');

        if (d.alarm_raised) showToast('Yüksek risk: alarm açıldı, doktor onayı bekleniyor', 'warning');

        await openPatient(currentPatientId, { keepScroll: true });
        renderAssessment(d);
        loadPatients();
        loadAlarms();
    } catch (e) {
        console.error(e);
        showToast(errorText(e), 'error');
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

function closeAddPatientModal() { closeModal('modal-add-patient'); }
function closeModal(id) { document.getElementById(id).hidden = true; }

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
        const d = await api('/api/patients', { method: 'POST', body: payload });
        showToast(`${payload.name} kabul edildi`, 'success');
        closeAddPatientModal();
        document.getElementById('add-patient-form').reset();
        await loadPatients();
        openPatient(d.patient_id);
    } catch (e) {
        showToast(errorText(e), 'error');
    }
}

let outcomeMode = 'discharge';   // 'discharge' | 'edit'

function dischargePatient() { showOutcomeModal('discharge'); }
function editOutcome() { showOutcomeModal('edit'); }

function showOutcomeModal(mode) {
    outcomeMode = mode;
    const oc = currentPatient.outcome;
    document.getElementById('outcome-title').textContent = mode === 'discharge' ? 'Taburcu ve sonuç' : 'Sonucu düzenle';
    document.getElementById('outcome-desc').textContent = mode === 'discharge'
        ? `${currentPatient.name} taburcu edilip arşive alınacak. Açık alarmları kapatılır.`
        : `${currentPatient.name} için kayıtlı sonucu güncelleyin (ör. kültür sonucu sonradan geldiyse).`;
    document.getElementById('outcome-submit').textContent = mode === 'discharge' ? 'Taburcu et' : 'Kaydet';
    document.querySelectorAll('input[name="outcome-sepsis"]').forEach(r => { r.checked = !!oc && oc.sepsis === r.value; });
    const last = currentHourly[currentHourly.length - 1];
    document.getElementById('outcome-onset').value = oc && oc.onset_hour != null ? oc.onset_hour : '';
    document.getElementById('outcome-onset-unknown').checked = !!oc && oc.sepsis === 'yes' && oc.onset_hour == null;
    document.getElementById('outcome-onset').placeholder = last ? `0–${last.hour}` : '';
    document.getElementById('outcome-note').value = oc && oc.note ? oc.note : '';
    toggleOnset();
    document.getElementById('modal-outcome').hidden = false;
}

function toggleOnset() {
    const sel = document.querySelector('input[name="outcome-sepsis"]:checked');
    document.getElementById('outcome-onset-wrap').hidden = !(sel && sel.value === 'yes');
    const unknown = document.getElementById('outcome-onset-unknown').checked;
    document.getElementById('outcome-onset').disabled = unknown;
    document.getElementById('outcome-onset-unknown-help').hidden = !unknown;
}

async function submitOutcome(event) {
    event.preventDefault();
    const sel = document.querySelector('input[name="outcome-sepsis"]:checked');
    if (!sel) return showToast('Sepsis sonucunu seçin', 'warning');
    const onset = document.getElementById('outcome-onset').value;
    const onsetUnknown = sel.value === 'yes' && document.getElementById('outcome-onset-unknown').checked;
    if (sel.value === 'yes' && !onsetUnknown && onset === '') return showToast('Başlangıç saatini girin veya “bilinmiyor” işaretleyin', 'warning');
    const outcome = { sepsis: sel.value, onset_unknown: onsetUnknown,
                      onset_hour: sel.value === 'yes' && !onsetUnknown ? parseInt(onset, 10) : null,
                      note: document.getElementById('outcome-note').value.trim() };
    try {
        if (outcomeMode === 'discharge') {
            await api(`/api/patients/${currentPatientId}/discharge`, { method: 'POST', body: { outcome } });
            showToast('Hasta taburcu edildi ve arşive alındı', 'success');
        } else {
            await api(`/api/patients/${currentPatientId}/outcome`, { method: 'PUT', body: { outcome } });
            showToast('Sonuç güncellendi', 'success');
        }
        closeModal('modal-outcome');
        loadPatients();
        loadAlarms();
        openPatient(currentPatientId, { keepScroll: true });
    } catch (e) {
        showToast(errorText(e), 'error');
    }
}

const OUTCOME_LABELS = { yes: 'Sepsis: Evet', no: 'Sepsis: Hayır', unknown: 'Sepsis: Belirsiz' };

function onsetText(oc) {
    if (oc.sepsis !== 'yes') return '';
    return oc.onset_hour != null ? ` (saat ${oc.onset_hour})` : ' (saat bilinmiyor)';
}

async function readmitPatient() {
    if (!confirm(`"${currentPatient.name}" yeniden kabul edilsin mi?`)) return;
    try {
        await api(`/api/patients/${currentPatientId}/readmit`, { method: 'POST' });
        showToast('Hasta yeniden aktif', 'success');
        loadPatients();
        openPatient(currentPatientId, { keepScroll: true });
    } catch (e) {
        showToast(errorText(e), 'error');
    }
}

async function deletePatient(id) {
    const p = [...allPatients, ...archivedPatients].find(x => x.id === id) || currentPatient;
    if (!confirm(`"${p ? p.name : id}" ve tüm ölçümleri, tahminleri ve alarmları KALICI olarak silinsin mi?\n\nBu işlem geri alınamaz; denetim kaydına yazılır.`)) return;
    try {
        await api(`/api/patients/${id}`, { method: 'DELETE' });
        showToast('Hasta kalıcı olarak silindi', 'info');
        if (currentPatientId === id) {
            currentPatientId = null;
            document.getElementById('nav-patient-detail').hidden = true;
            switchView('patients');
        }
        loadPatients();
        loadAlarms();
    } catch (e) {
        showToast(errorText(e), 'error');
    }
}

function deleteCurrentPatient() { if (currentPatientId) deletePatient(currentPatientId); }

// ===================== PASSWORD =====================
let passwordTarget = null;   // null = kendi şifrem, aksi halde kullanıcı id (yönetici sıfırlaması)

function showPasswordModal(userId = null, username = '') {
    passwordTarget = userId;
    document.getElementById('pw-modal-title').textContent = userId ? 'Şifre sıfırla' : 'Şifre değiştir';
    document.getElementById('pw-modal-desc').textContent = userId
        ? `${username} için yeni geçici şifre. Kullanıcının açık oturumları kapatılır.` : '';
    document.getElementById('pw-current-wrap').hidden = !!userId;
    document.getElementById('pw-current').value = '';
    document.getElementById('pw-new').value = '';
    document.getElementById('modal-password').hidden = false;
    document.getElementById(userId ? 'pw-new' : 'pw-current').focus();
}

async function submitPassword(event) {
    event.preventDefault();
    const newPassword = document.getElementById('pw-new').value;
    try {
        if (passwordTarget) {
            await api(`/api/users/${passwordTarget}`, { method: 'PATCH', body: { password: newPassword } });
        } else {
            await api('/api/auth/change-password', { method: 'POST',
                body: { current_password: document.getElementById('pw-current').value, new_password: newPassword } });
        }
        closeModal('modal-password');
        showToast('Şifre güncellendi', 'success');
        if (passwordTarget) loadUsers();
    } catch (e) {
        showToast(errorText(e), 'error');
    }
}

// ===================== ADMIN: USERS =====================
const PERMISSION_LABELS = {
    'patients.view': 'Hasta listesi ve risk görme', 'patients.create': 'Hasta kabul',
    'observations.create': 'Saatlik veri girişi', 'alarms.view': 'Alarmları görme',
    'alarms.acknowledge': 'Alarm onayı ve klinik not', 'patients.discharge': 'Taburcu / yeniden kabul',
    'outcomes.label': 'Sepsis sonucunu işaretleme (veri seti etiketi)',
    'history.view': 'Hasta kayıt geçmişi', 'model.view': 'Model bilgisi', 'patients.delete': 'Kalıcı silme',
    'dataset.export': 'Veri setini dışa aktarma',
    'users.manage': 'Kullanıcı yönetimi', 'audit.view': 'Denetim kayıtları ve veritabanı',
};
const ROLE_PERMISSIONS = {
    nurse: ['patients.view', 'patients.create', 'observations.create', 'alarms.view', 'history.view', 'model.view'],
    doctor: ['patients.view', 'patients.create', 'observations.create', 'alarms.view', 'alarms.acknowledge', 'patients.discharge', 'outcomes.label', 'history.view', 'model.view'],
    admin: ['patients.view', 'alarms.view', 'history.view', 'model.view', 'patients.delete', 'users.manage', 'audit.view', 'dataset.export'],
};

async function loadUsers() {
    if (!can('users.manage')) return;
    const tbody = document.getElementById('users-body');
    try {
        const users = (await api('/api/users')).users;
        tbody.innerHTML = users.map(u => {
            const locked = u.locked_until && parseTs(u.locked_until) > new Date();
            const self = u.id === currentUser.id;
            return `
            <tr>
                <td><div class="person"><div class="avatar">${escapeHtml(initials(u.full_name))}</div><div><strong>${escapeHtml(u.full_name)}</strong>${self ? '<span>siz</span>' : ''}</div></div></td>
                <td class="num">${escapeHtml(u.username)}</td>
                <td>${self ? `<span class="role-badge role-${u.role}">${u.role_label}</span>` :
                    `<select class="role-select" onchange="updateUser(${u.id}, { role: this.value })">
                        ${['nurse', 'doctor', 'admin'].map(r => `<option value="${r}" ${r === u.role ? 'selected' : ''}>${roleLabel(r)}</option>`).join('')}
                     </select>`}</td>
                <td>${locked ? '<span class="badge high">Kilitli</span>' : u.is_active ? '<span class="badge very-low">Aktif</span>' : '<span class="badge none">Devre dışı</span>'}</td>
                <td class="small muted">${u.last_login ? formatDateTime(u.last_login) : 'Hiç giriş yapmadı'}</td>
                <td class="row-actions">
                    ${locked ? `<button class="btn btn-ghost btn-sm" onclick="updateUser(${u.id}, { unlock: true })">Kilidi aç</button>` : ''}
                    <button class="btn btn-ghost btn-sm" onclick="showPasswordModal(${u.id}, '${escapeHtml(u.username)}')">Şifre sıfırla</button>
                    ${self ? '' : `<button class="btn btn-ghost btn-sm" onclick="updateUser(${u.id}, { is_active: ${!u.is_active} })">${u.is_active ? 'Devre dışı bırak' : 'Etkinleştir'}</button>`}
                </td>
            </tr>`;
        }).join('');
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6"><div class="empty"><p>${escapeHtml(e.message)}</p></div></td></tr>`;
    }
    const roles = ['nurse', 'doctor', 'admin'];
    document.getElementById('permissions-table').innerHTML = `
        <table class="table perm-table">
            <thead><tr><th>Yetki</th>${roles.map(r => `<th>${roleLabel(r)}</th>`).join('')}</tr></thead>
            <tbody>${Object.entries(PERMISSION_LABELS).map(([p, label]) => `
                <tr><td>${label}</td>${roles.map(r => `<td>${ROLE_PERMISSIONS[r].includes(p) ? '<span class="yes">✓</span>' : '<span class="no">—</span>'}</td>`).join('')}</tr>`).join('')}
            </tbody>
        </table>`;
}

async function updateUser(id, changes) {
    try {
        await api(`/api/users/${id}`, { method: 'PATCH', body: changes });
        showToast('Kullanıcı güncellendi', 'success');
    } catch (e) {
        showToast(errorText(e), 'error');
    }
    loadUsers();
}

function showAddUserModal() {
    document.getElementById('modal-add-user').hidden = false;
    document.getElementById('new-user-name').focus();
}

async function addUser(event) {
    event.preventDefault();
    const body = {
        full_name: document.getElementById('new-user-name').value.trim(),
        username: document.getElementById('new-user-username').value.trim(),
        role: document.getElementById('new-user-role').value,
        password: document.getElementById('new-user-password').value,
    };
    try {
        await api('/api/users', { method: 'POST', body });
        showToast(`${body.full_name} eklendi`, 'success');
        closeModal('modal-add-user');
        event.target.reset();
        loadUsers();
    } catch (e) {
        showToast(errorText(e), 'error');
    }
}

// ===================== ADMIN: AUDIT =====================
async function loadAudit(more = false) {
    if (!can('audit.view')) return;
    auditOffset = more ? auditOffset + 100 : 0;
    const params = new URLSearchParams({ limit: 100, offset: auditOffset });
    const user = document.getElementById('audit-user').value.trim();
    const action = document.getElementById('audit-action').value;
    if (user) params.set('username', user);
    if (action) params.set('action', action);
    if (document.getElementById('audit-failed').checked) params.set('failed', '1');
    const tbody = document.getElementById('audit-body');
    try {
        const d = await api(`/api/audit-logs?${params}`);
        const rows = d.logs.map(l => `
            <tr class="${l.success ? '' : 'row-failed'}">
                <td class="small num">${formatDateTime(l.timestamp)}</td>
                <td>${escapeHtml(l.username || 'Sistem')}${l.role && l.role !== 'system' ? `<br><span class="role-badge role-${l.role}">${roleLabel(l.role)}</span>` : ''}</td>
                <td><strong>${ACTION_LABELS[l.action] || l.action}</strong>${l.success ? '' : ' <span class="badge high">Başarısız</span>'}<br><span class="muted small num">${l.action}</span></td>
                <td>${l.patient_id ? (l.patient_name ? `<a class="link-plain" onclick="openPatient(${l.patient_id})">${escapeHtml(l.patient_name)}</a>` : `<span class="muted">#${l.patient_id} (silindi)</span>`) : '—'}</td>
                <td class="small">${escapeHtml(describeEvent(l.action, l.details))}</td>
                <td class="small muted num">${escapeHtml(l.ip_address || '')}</td>
            </tr>`).join('');
        if (more) tbody.insertAdjacentHTML('beforeend', rows); else tbody.innerHTML = rows || '<tr><td colspan="6"><div class="empty"><p>Kayıt yok</p></div></td></tr>';
        const shown = Math.min(auditOffset + 100, d.total);
        document.getElementById('audit-count').textContent = `${shown} / ${d.total} kayıt · kayıtlar değiştirilemez`;
        document.getElementById('audit-more').hidden = shown >= d.total;
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6"><div class="empty"><p>${escapeHtml(e.message)}</p></div></td></tr>`;
    }
}

// ===================== ADMIN: DATABASE =====================
const SCHEMA_DOC = [
    ['users', 'Kullanıcılar', ['id PK', 'username UNIQUE', 'password_hash', 'full_name', 'role CHECK(admin|doctor|nurse)', 'is_active', 'failed_login_attempts', 'locked_until', 'last_login']],
    ['sessions', 'Oturumlar', ['token_hash PK (SHA-256)', 'user_id → users', 'ip_address', 'last_activity']],
    ['patients', 'Hastalar', ['id PK', 'patient_id UNIQUE', 'name', 'age CHECK(0–120)', 'gender', 'status CHECK(active|discharged)', 'created_by → users', 'discharged_by → users']],
    ['observations', 'Saatlik kayıtlar', ['id PK', 'patient_id → patients', 'hour CHECK(≥0)', 'entered_by → users', 'UNIQUE(patient_id, hour)']],
    ['measurements', 'Ölçümler (her değer bir satır)', ['observation_id → observations', 'variable', 'value', 'PK(observation_id, variable)']],
    ['predictions', 'Model tahminleri', ['observation_id → observations (1–1)', 'model_version', 'score', 'calibrated_probability', 'risk_level', 'reliability', 'assessment (JSON)']],
    ['alarms', 'Alarmlar', ['patient_id → patients', 'observation_id → observations', 'risk_level', 'status CHECK(open|acknowledged)', 'acknowledged_by → users', 'note']],
    ['outcomes', 'Sonuç (veri seti etiketi)', ['patient_id PK + → patients (1–1)', 'sepsis CHECK(yes|no|unknown)', 'onset_hour', 'labeled_by → users', 'CHECK: başlangıç saati yalnızca sepsiste']],
    ['audit_logs', 'Denetim kayıtları', ['user_id → users', 'username, role', 'action', 'patient_id', 'details (JSON)', 'ip_address', 'success', 'UPDATE/DELETE tetikleyiciyle engelli']],
];

async function loadDatabase() {
    if (!can('audit.view')) return;
    const body = document.getElementById('database-body');
    try {
        const d = await api('/api/admin/db-stats');
        body.innerHTML = `
            <div class="card">
                <div class="card-head"><h2>Tablolar</h2><span class="legend-note">${escapeHtml(d.database)} · şema sürümü ${d.schema_version} · SQLite</span></div>
                <div class="metrics">${Object.entries(d.tables).map(([t, n]) => `
                    <div class="metric"><div class="metric-label num">${t}</div><div class="metric-value">${n.toLocaleString('tr-TR')}</div></div>`).join('')}</div>
            </div>
            <div class="card">
                <div class="card-head">
                    <h2>Yeniden eğitim için veri seti</h2>
                    ${can('dataset.export') ? `<a class="btn btn-primary btn-sm" href="/api/admin/dataset.csv" download>CSV indir</a>` : ''}
                </div>
                <div class="metrics">
                    <div class="metric"><div class="metric-label">Sepsis (Evet)</div><div class="metric-value bad">${d.dataset.sepsis_yes}</div></div>
                    <div class="metric"><div class="metric-label">Sepsis değil (Hayır)</div><div class="metric-value good">${d.dataset.sepsis_no}</div></div>
                    <div class="metric"><div class="metric-label">Sepsis, saat bilinmiyor (alınmaz)</div><div class="metric-value">${d.dataset.sepsis_no_onset}</div></div>
                    <div class="metric"><div class="metric-label">Belirsiz (alınmaz)</div><div class="metric-value">${d.dataset.sepsis_unknown}</div></div>
                    <div class="metric"><div class="metric-label">Sonuç girilmemiş</div><div class="metric-value warn">${d.dataset.unlabeled}</div></div>
                    <div class="metric"><div class="metric-label">Veri setindeki saat</div><div class="metric-value">${d.dataset.exportable_hours.toLocaleString('tr-TR')}</div></div>
                </div>
                <div class="prose"><p>Taburcu edilen ve doktorun sonucunu işaretlediği hastalar, <strong>kimlik bilgisi olmadan</strong> PhysioNet 2019 formatında dışa aktarılır (ad ve protokol no yerine L00001 gibi sıra no). Etiket kuralı yarışmayla aynı: klinik başlangıçtan 6 saat önce <span class="num">SepsisLabel = 1</span>. Dosya <span class="num">scripts/train_v3.py --input</span> ile modeli yeniden eğitmek için kullanılabilir; komut satırından: <span class="num">py -3.10 scripts/veri_seti_disa_aktar.py</span></p></div>
            </div>
            <div class="card">
                <div class="card-head"><h2>İlişkiler</h2></div>
                <pre class="er">users ──< sessions
  │
  ├──< audit_logs          (değiştirilemez)
  │
patients ──< observations ──< measurements
   │              │
   │              └──── predictions   (1–1)
   ├──< alarms
   └──── outcomes     (1–1, doktorun işaretlediği sonuç)</pre>
            </div>
            <div class="card">
                <div class="card-head"><h2>Şema</h2><span class="legend-note">PK: birincil anahtar · → : yabancı anahtar</span></div>
                <div class="schema-grid">${SCHEMA_DOC.map(([t, label, cols]) => `
                    <div class="schema-table"><div class="schema-name"><span class="num">${t}</span><span class="muted small">${label}</span></div>
                    <ul>${cols.map(c => `<li class="num">${escapeHtml(c)}</li>`).join('')}</ul></div>`).join('')}
                </div>
                <div class="prose"><p><strong>Bütünlük:</strong> yabancı anahtarlar açık (hasta silinince ölçüm, tahmin ve alarmları CASCADE ile silinir), CHECK kısıtları geçersiz rol, yaş ve durumu reddeder, şifreler tuzlu hash olarak saklanır, denetim kayıtları tetikleyicilerle değiştirilemez.</p></div>
            </div>`;
    } catch (e) {
        body.innerHTML = `<div class="card"><div class="empty"><p>${escapeHtml(e.message)}</p></div></div>`;
    }
}

// ===================== MODEL INFO =====================
async function loadModelInfo() {
    const body = document.getElementById('model-info-body');
    try {
        modelInfo = await api('/api/model-info');
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
                ${modelInfo.feature_set === 'v4' ? metric('Özellik', modelInfo.features.length) : metric('Pencere × özellik', `${modelInfo.window_size} × ${modelInfo.features.length}`)}
                ${m.n_trees ? metric('Ağaç / ağırlık', `${m.n_trees} / ${(m.params || 0).toLocaleString('tr-TR')}`) : metric('Parametre', (m.params || 0).toLocaleString('tr-TR'))}
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
        ${renderCrossValidation(modelInfo.cross_validation)}
        ${renderExternalValidation(modelInfo.external_validation)}
        ${renderClinicalComparison(modelInfo.clinical_comparison)}
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
    const c = st.components_test;
    const rows = c ? [
        [`Hibrit ${c.hybrid_v4 ? 'v5' : 'v4'} (aktif)`, 'Hasta bazlı', c.hybrid.roc_auc, c.hybrid.pr_auc, `LightGBM ${Math.round(st.ensemble_weight_lightgbm * 100)}% + sinir ağı ${Math.round((1 - st.ensemble_weight_lightgbm) * 100)}%`],
        ['└ LightGBM dalı', 'Hasta bazlı', c.lightgbm.roc_auc, c.lightgbm.pr_auc, c.hybrid_v4 ? 'Ayarlı, monoton kısıtlı, 3 seed' : 'Zaman serisi özellikleri, 3 seed'],
        ['└ Sinir ağı dalı (MLP)', 'Hasta bazlı', c.mlp.roc_auc, c.mlp.pr_auc, 'Aynı özellikler, 3 gizli katman'],
        c.hybrid_v4 ? ['Hibrit v4', 'Hasta bazlı', c.hybrid_v4.roc_auc, c.hybrid_v4.pr_auc, 'Önceki sürüm: LightGBM 70% + sinir ağı 30%'] : null,
        ['GRU v3', 'Hasta bazlı', c.gru_v3.roc_auc, c.gru_v3.pr_auc, 'Son 6 saatin ham değerleri'],
    ] : [
        ['GRU v3 (aktif)', 'Hasta bazlı', t.roc_auc, t.pr_auc, 'İleri doldurma + ölçüm maskeleri'],
        lg ? ['LightGBM (karşılaştırma)', 'Hasta bazlı, aynı hastalar', lg.roc_auc, lg.pr_auc, 'Aynı özellikler, ağaç tabanlı'] : null,
    ];
    rows.push(
        ['Optimized v1 (eski)', 'Hasta bazlı', 0.809, 0.089, 'Medyan doldurma; vital bulgulara tepkisiz'],
        ['GRU v23 (ilk sürüm)', 'Rastgele pencere', 0.880, 0.180, 'Aynı hasta hem eğitimde hem testte → iyimser'],
    );
    const shown = rows.filter(Boolean);
    return `
        <div class="card">
            <div class="card-head"><h2>Model karşılaştırması</h2></div>
            <div class="table-wrap"><table class="table">
                <thead><tr><th>Model</th><th>Test bölmesi</th><th>ROC-AUC</th><th>PR-AUC</th><th>Not</th></tr></thead>
                <tbody>${shown.map(r => `<tr><td><strong>${r[0]}</strong></td><td>${r[1]}</td><td class="num">${r[2].toFixed(3)}</td><td class="num">${r[3].toFixed(3)}</td><td class="muted small">${r[4]}</td></tr>`).join('')}</tbody>
            </table></div>
            <div class="prose"><p>İlk sürümün 0.88'i, veri saat saat rastgele bölündüğü için aynı hastanın bir saati eğitimde, sonraki saati testte bulunarak elde edildi. Yaş + cinsiyet + yatış süresi hastaların %93'ünü tek başına tanımladığından model hastayı “tanıyabiliyordu”. Hasta bazlı bölmede hiçbir test hastası eğitimde görülmez; gerçekçi değer budur.</p></div>
        </div>`;
}

function renderCrossValidation(cv) {
    if (!cv) return '';
    const s = cv.summary;
    const row = (label, key) => `<tr><td><strong>${label}</strong></td>
        <td class="num">${s[key + '_roc_auc'].mean.toFixed(3)} ± ${s[key + '_roc_auc'].std.toFixed(3)}</td>
        <td class="num">${s[key + '_pr_auc'].mean.toFixed(3)} ± ${s[key + '_pr_auc'].std.toFixed(3)}</td>
        <td class="num muted">${s[key + '_roc_auc'].min.toFixed(3)} – ${s[key + '_roc_auc'].max.toFixed(3)}</td></tr>`;
    return `
        <div class="card">
            <div class="card-head"><h2>${cv.k} katlı çapraz doğrulama</h2><span class="legend-note">${cv.n_patients.toLocaleString('tr-TR')} hasta, hasta bazlı · ortalama ± standart sapma</span></div>
            <div class="table-wrap"><table class="table">
                <thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>ROC-AUC aralığı</th></tr></thead>
                <tbody>${row('Hibrit', 'hybrid')}${row('LightGBM', 'lightgbm')}${row('Sinir ağı (MLP)', 'mlp')}</tbody>
            </table></div>
            <div class="prose"><p>Her hasta bir kez test katında yer alır. Düşük standart sapma, sonucun tek bir veri bölmesine bağlı olmadığını gösterir.</p></div>
        </div>`;
}

function renderExternalValidation(ev) {
    if (!ev) return '';
    const names = { 'dış': 'Dış: yalnız diğer hastane', 'iç': 'İç: kendi verisi', 'birleşik': 'Birleşik: diğer + kendi' };
    const order = ['dış', 'iç', 'birleşik'];
    const rows = Object.entries(ev.results).map(([target, res]) => Object.entries(res).sort((x, y) => order.indexOf(x[0]) - order.indexOf(y[0])).map(([name, r], i) => `
        <tr>${i === 0 ? `<td rowspan="${Object.keys(res).length}"><strong>Hastane ${target}</strong><div class="muted small">${r.test_patients.toLocaleString('tr-TR')} test hastası</div></td>` : ''}
            <td>${names[name] || escapeHtml(name)}</td>
            <td class="num">${r.train_patients.toLocaleString('tr-TR')}</td>
            <td class="num">${name === 'dış' ? '' : '<strong>'}${r.roc_auc.toFixed(3)}${name === 'dış' ? '' : '</strong>'}</td>
            <td class="num">${r.pr_auc.toFixed(3)}</td><td class="num">${r.utility.toFixed(3)}</td>
            <td class="num">${pct(r.screen_sensitivity, 0)} / ${pct(r.screen_specificity, 0)}</td></tr>`).join('')).join('');
    return `
        <div class="card">
            <div class="card-head"><h2>Hastaneler arası test</h2><span class="muted small">PhysioNet 2019: hastane A ve B</span></div>
            <div class="table-wrap"><table class="table">
                <thead><tr><th>Test edilen</th><th>Eğitim</th><th>Eğitim hastası</th><th>ROC-AUC</th><th>PR-AUC</th><th>Utility</th><th>Tarama eşiği duyarlılık / özgüllük</th></tr></thead>
                <tbody>${rows}</tbody>
            </table></div>
            <div class="prose"><p>Model hiç görmediği hastaneye götürüldüğünde ROC-AUC 0.03–0.07 düşüyor ve eğitim hastanesinde seçilen eşik taşınmıyor (B'de tarama duyarlılığı %80 yerine %59). Aynı hastanenin kendi verisiyle eğitilince performans geri geliyor. Sonuç: <strong>yeni bir hastanede kullanmadan önce o hastanenin verisiyle doğrulama ve yeniden eğitim şart</strong>; uygulamadaki taburcu sonucu → anonim veri seti döngüsü bunun için var. Eşikler her modelin kendi eğitim hastanesinin validasyonunda seçildi; tek koşu olduğu için 0.01'lik farklar anlamlı değildir.</p></div>
        </div>`;
}

function renderClinicalComparison(cc) {
    if (!cc) return '';
    const rows = Object.entries(cc.models).sort((x, y) => y[1].utility_test - x[1].utility_test).map(([name, m]) => `
        <tr><td><strong>${escapeHtml(name)}</strong></td>
            <td class="num">${m.roc_auc.toFixed(3)}</td><td class="num">${m.pr_auc.toFixed(3)}</td>
            <td class="num">${m.utility_test.toFixed(3)}</td>
            <td class="num">${pct(m.at_utility_threshold_sensitivity ?? m.at_cutoff_sensitivity, 0)} / ${pct(m.at_utility_threshold_specificity ?? m.at_cutoff_specificity, 0)}</td>
            <td class="muted small">${m.cutoff ? 'kesim ' + escapeHtml(m.cutoff) : 'Utility eşiği (val.) ' + m.utility_threshold_val.toFixed(3)}</td></tr>`).join('');
    return `
        <div class="card">
            <div class="card-head"><h2>Klinik skorlarla karşılaştırma ve PhysioNet Utility</h2></div>
            <div class="table-wrap"><table class="table">
                <thead><tr><th>Yöntem</th><th>ROC-AUC</th><th>PR-AUC</th><th>Utility</th><th>Duyarlılık / özgüllük</th><th>Alarm kuralı</th></tr></thead>
                <tbody>${rows}</tbody>
            </table></div>
            <div class="prose"><p><strong>Utility</strong>, PhysioNet 2019 yarışmasının resmi metriğidir: sepsisten 12–6 saat önceki alarmı ödüllendirir, geç veya yanlış alarmı cezalandırır (1 = mükemmel, 0 = hiç alarm vermemek). Yarışma kazananı 0.36'yı gizli test setinde aldı (A ve B'nin yayımlanmamış hastaları + hiç görülmemiş üçüncü hastane); bizim değer aynı iki hastanenin ayrılmış test hastalarında ölçüldüğü için <strong>doğrudan karşılaştırılamaz</strong>. Görülmemiş hastanede bizim Utility'miz 0.28–0.36 (hastaneler arası test). qSOFA ve NEWS2 bilinç (GKS) verisi olmadığı için kısmidir.</p></div>
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
