// ============================================================================
// GLOBAL DEĞİŞKENLER
// ============================================================================

const API_URL = 'http://localhost:5000';
let currentPatientId = null;
let riskChart = null;
let currentUser = null;

// ============================================================================
// SAYFA YÜKLENDİĞİNDE
// ============================================================================

document.addEventListener('DOMContentLoaded', async function () {
    // Authentication kontrolü
    await checkAuth();

    // Kullanıcı bilgisini yükle ve göster
    await loadUserInfo();

    // Ana içeriği yükle
    loadPatients();
    updateNextHour();
});

// ============================================================================
// AUTHENTICATION KONTROLÜ
// ============================================================================

async function checkAuth() {
    try {
        const response = await fetch(`${API_URL}/api/auth/me`);
        const data = await response.json();

        if (!data.success) {
            // Login olmamış, login sayfasına yönlendir
            window.location.href = '/login.html';
            return false;
        }

        currentUser = data.user;
        return true;
    } catch (error) {
        console.error('Auth check error:', error);
        window.location.href = '/login.html';
        return false;
    }
}

async function loadUserInfo() {
    if (currentUser) {
        // Kullanıcı adını header'da göster
        const userInfoDiv = document.getElementById('user-info');
        if (userInfoDiv) {
            userInfoDiv.innerHTML = `
                <span class="user-name">👤 ${currentUser.username}</span>
                <button class="btn btn-secondary btn-sm" onclick="logout()">Çıkış</button>
            `;
        }
    }
}

function logout() {
    fetch(`${API_URL}/api/auth/logout`, { method: 'POST' })
        .then(() => {
            window.location.href = '/login.html';
        })
        .catch(err => {
            console.error('Logout error:', err);
            window.location.href = '/login.html';
        });
}

// ============================================================================
// HASTA LİSTESİ FONKSİYONLARI
// ============================================================================

async function loadPatients() {
    try {
        const response = await fetch(`${API_URL}/api/patients`);
        const data = await response.json();

        if (data.success) {
            displayPatients(data.patients);
        } else {
            showToast('Hastalar yüklenemedi', 'error');
        }
    } catch (error) {
        console.error('Hasta listesi hatası:', error);
        showToast('Bağlantı hatası', 'error');
    }
}

function displayPatients(patients) {
    const container = document.getElementById('patients-container');

    if (patients.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <p>👤 Henüz hasta kaydı yok</p>
                <button class="btn btn-primary" onclick="showAddPatientModal()">
                    İlk Hastayı Ekle
                </button>
            </div>
        `;
        return;
    }

    container.innerHTML = patients.map(patient => `
        <div class="patient-card" onclick="showPatientDetail(${patient.id})">
            <div class="patient-card-header">
                <h3>${patient.name}</h3>
                <span class="patient-id">${patient.patient_id}</span>
            </div>
            <div class="patient-card-body">
                <p><strong>Yaş:</strong> ${patient.age || 'Belirtilmemiş'}</p>
                <p><strong>Cinsiyet:</strong> ${patient.gender || 'Belirtilmemiş'}</p>
                <p><strong>Toplam Saat:</strong> ${patient.total_hours || 0}</p>
            </div>
            <div class="patient-card-footer">
                ${patient.latest_risk_level ? `
                    <div class="risk-badge risk-${patient.latest_risk_level.toLowerCase().replace(' ', '-')}">
                        ${patient.latest_risk_level}
                    </div>
                    <div class="risk-score">
                        Risk: ${(patient.latest_prediction * 100).toFixed(1)}%
                    </div>
                ` : `
                    <div class="text-muted">Veri bekleniyor...</div>
                `}
            </div>
        </div>
    `).join('');
}

// ============================================================================
// HASTA EKLEME
// ============================================================================

function showAddPatientModal() {
    document.getElementById('add-patient-modal').classList.add('active');
}

function closeAddPatientModal() {
    document.getElementById('add-patient-modal').classList.remove('active');
    document.getElementById('add-patient-form').reset();
}

async function addPatient(event) {
    event.preventDefault();

    const patientData = {
        patient_id: document.getElementById('new-patient-id').value,
        name: document.getElementById('new-patient-name').value,
        age: parseInt(document.getElementById('new-patient-age').value) || null,
        gender: document.getElementById('new-patient-gender').value || null,
        admission_time: new Date().toISOString()
    };

    try {
        const response = await fetch(`${API_URL}/api/patients`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(patientData)
        });

        const data = await response.json();

        if (data.success) {
            showToast('✅ Hasta başarıyla eklendi', 'success');
            closeAddPatientModal();
            loadPatients();
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Hasta ekleme hatası:', error);
        showToast('Bağlantı hatası', 'error');
    }
}

// ============================================================================
// HASTA DETAY
// ============================================================================

async function showPatientDetail(patientId) {
    currentPatientId = patientId;

    try {
        const response = await fetch(`${API_URL}/api/patients/${patientId}`);
        const data = await response.json();

        if (data.success) {
            displayPatientDetail(data.patient, data.hourly_data);

            // Görünüm değiştir
            document.getElementById('patient-list-view').classList.remove('active');
            document.getElementById('patient-detail-view').classList.add('active');
        } else {
            showToast('Hasta detayları yüklenemedi', 'error');
        }
    } catch (error) {
        console.error('Hasta detay hatası:', error);
        showToast('Bağlantı hatası', 'error');
    }
}

function displayPatientDetail(patient, hourlyData) {
    // Hasta bilgilerini göster
    document.getElementById('patient-name').textContent = patient.name;
    document.getElementById('patient-meta').textContent =
        `ID: ${patient.patient_id} | Yaş: ${patient.age || '?'} | Cinsiyet: ${patient.gender || '?'}`;

    // Risk trend grafiğini çiz
    drawRiskChart(hourlyData);

    // Saatlik geçmişi göster
    displayHourlyHistory(hourlyData);

    // Sonraki saat değerini ayarla
    if (hourlyData.length > 0) {
        const lastHour = Math.max(...hourlyData.map(h => h.hour));
        document.getElementById('hour-input').value = lastHour + 1;
    } else {
        document.getElementById('hour-input').value = 1;
    }
}

function drawRiskChart(hourlyData) {
    const ctx = document.getElementById('risk-chart').getContext('2d');

    // Eski grafiği temizle
    if (riskChart) {
        riskChart.destroy();
    }

    if (hourlyData.length === 0) {
        ctx.font = '16px Arial';
        ctx.fillStyle = '#999';
        ctx.textAlign = 'center';
        ctx.fillText('Veri bekleniyor...', ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    }

    const hours = hourlyData.map(h => `Saat ${h.hour}`);
    const predictions = hourlyData.map(h => h.prediction * 100);

    riskChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [{
                label: 'Sepsis Riski (%)',
                data: predictions,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 8
            }, {
                label: 'Eşik Değeri (17.99%)',
                data: Array(hours.length).fill(17.99),
                borderColor: '#ef4444',
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function (value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

function displayHourlyHistory(hourlyData) {
    const container = document.getElementById('hourly-history');

    if (hourlyData.length === 0) {
        container.innerHTML = '<p class="text-muted">Henüz saatlik veri yok</p>';
        return;
    }

    container.innerHTML = hourlyData.map(data => {
        const riskClass = data.risk_level.toLowerCase().replace(' ', '-');
        const vs = data.vital_signs;

        return `
            <div class="hourly-card">
                <div class="hourly-header">
                    <h4>Saat ${data.hour}</h4>
                    <div class="risk-badge risk-${riskClass}">
                        ${data.risk_level}
                    </div>
                </div>
                <div class="hourly-body">
                    <div class="vital-grid">
                        ${Object.entries(vs).map(([key, value]) => `
                            <div class="vital-item">
                                <span class="vital-label">${key}:</span>
                                <span class="vital-value">${value !== null ? value : 'N/A'}</span>
                            </div>
                        `).join('')}
                    </div>
                    <div class="prediction-info">
                        <strong>Risk Skoru:</strong> ${(data.prediction * 100).toFixed(2)}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ============================================================================
// SAATLİK VERİ EKLEME
// ============================================================================

async function addHourlyData(event) {
    event.preventDefault();

    const hour = parseInt(document.getElementById('hour-input').value);

    // Form verilerini topla - 34 klinik parametre + 22 yeni biomarker = 56 toplam
    const vitalSigns = {};
    const fields = [
        // Vital Signs
        'HR', 'O2Sat', 'Temp', 'Resp',
        // Blood Pressure
        'SBP', 'DBP', 'MAP',
        // Respiratory
        'EtCO2', 'FiO2', 'PaCO2', 'SaO2',
        // Acid-Base
        'pH', 'BaseExcess', 'HCO3',
        // Liver & Kidney
        'BUN', 'Creatinine', 'AST', 'Alkalinephos',
        'Bilirubin_total', 'Bilirubin_direct',
        // Electrolytes & Metabolites
        'Calcium', 'Chloride', 'Potassium', 'Magnesium',
        'Phosphate', 'Glucose', 'Lactate',
        // Hematology
        'WBC', 'Hct', 'Hgb', 'Platelets',
        // Coagulation & Cardiac
        'PTT', 'Fibrinogen', 'TroponinI',

        // ========== PHASE 2: NEW BIOMARKERS ==========
        // Sepsis Markers
        'PCT', 'CRP', 'Presepsin', 'IL6', 'IL1b',
        // Hematology Extended
        'ESR', 'MDW', 'MPV', 'RDW', 'Neutrophils', 'Lymphocytes',
        // Coagulation Extended
        'DDimer', 'PT', 'aPTT', 'INR',
        // Chemistry Extended
        'IonizedCalcium', 'Phosphorus', 'Albumin', 'Sodium'
    ];

    fields.forEach(field => {
        const value = document.getElementById(field).value;
        vitalSigns[field] = value ? parseFloat(value) : null;
    });

    const requestData = {
        hour: hour,
        vital_signs: vitalSigns
    };

    try {
        const response = await fetch(`${API_URL}/api/patients/${currentPatientId}/hourly-data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (data.success) {
            showToast(`✅ Saat ${hour} kaydedildi! Risk: ${(data.prediction * 100).toFixed(2)}%`, 'success');

            // Formu temizle
            event.target.reset();

            // Hasta detaylarını yeniden yükle
            showPatientDetail(currentPatientId);
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Saatlik veri ekleme hatası:', error);
        showToast('Bağlantı hatası', 'error');
    }
}

function updateNextHour() {
    // Bu fonksiyon hasta detayında otomatik çalışır
}

// ============================================================================
// HASTA SİLME
// ============================================================================

async function deleteCurrentPatient() {
    if (!currentPatientId) return;

    if (!confirm('Bu hastayı silmek istediğinizden emin misiniz?')) {
        return;
    }

    try {
        const response = await fetch(`${API_URL}/api/patients/${currentPatientId}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            showToast('✅ Hasta silindi', 'success');
            showPatientList();
            loadPatients();
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Hasta silme hatası:', error);
        showToast('Bağlantı hatası', 'error');
    }
}

// ============================================================================
// GÖRÜNÜM YÖNETİMİ
// ============================================================================

function showPatientList() {
    document.getElementById('patient-detail-view').classList.remove('active');
    document.getElementById('patient-list-view').classList.add('active');
    currentPatientId = null;
    loadPatients();
}

// ============================================================================
// YARDIMCI FONKSİYONLAR
// ============================================================================

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast toast-${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}
