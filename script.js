// ============================================================================
// GLOBAL DEĞİŞKENLER
// ============================================================================

const API_URL = 'http://localhost:5000';
let currentPatientId = null;
let riskChart = null;
let currentUser = null;

// ============================================================================
// VALIDATION RANGES
// ============================================================================

const VALIDATION_RANGES = {
    // Vital Signs (Core)
    'HR': { min: 40, max: 200, unit: 'bpm', name: 'Heart Rate' },
    'Temp': { min: 35, max: 42, unit: '°C', name: 'Temperature' },
    'SBP': { min: 60, max: 250, unit: 'mmHg', name: 'Systolic BP' },
    'DBP': { min: 30, max: 150, unit: 'mmHg', name: 'Diastolic BP' },
    'MAP': { min: 40, max: 180, unit: 'mmHg', name: 'Mean Arterial Pressure' },
    'Resp': { min: 8, max: 50, unit: '/min', name: 'Respiratory Rate' },
    'O2Sat': { min: 70, max: 100, unit: '%', name: 'Oxygen Saturation' },
    'EtCO2': { min: 10, max: 80, unit: 'mmHg', name: 'End-Tidal CO2' },

    // Lab - Hematology
    'WBC': { min: 1, max: 50, unit: 'K/µL', name: 'White Blood Cells' },
    'Platelets': { min: 20, max: 800, unit: 'K/µL', name: 'Platelets' },
    'Hgb': { min: 5, max: 20, unit: 'g/dL', name: 'Hemoglobin' },
    'Hct': { min: 15, max: 65, unit: '%', name: 'Hematocrit' },

    // Lab - Chemistry
    'Creatinine': { min: 0.3, max: 15, unit: 'mg/dL', name: 'Creatinine' },
    'BUN': { min: 3, max: 150, unit: 'mg/dL', name: 'Blood Urea Nitrogen' },
    'Glucose': { min: 30, max: 600, unit: 'mg/dL', name: 'Glucose' },
    'Lactate': { min: 0.5, max: 20, unit: 'mmol/L', name: 'Lactate' },
    'Bilirubin_total': { min: 0.1, max: 30, unit: 'mg/dL', name: 'Total Bilirubin' },
    'Bilirubin_direct': { min: 0, max: 15, unit: 'mg/dL', name: 'Direct Bilirubin' },

    // Lab - ABG
    'pH': { min: 6.8, max: 7.8, unit: '', name: 'pH' },
    'PaCO2': { min: 15, max: 100, unit: 'mmHg', name: 'PaCO2' },
    'PaO2': { min: 40, max: 500, unit: 'mmHg', name: 'PaO2' },
    'HCO3': { min: 10, max: 45, unit: 'mmol/L', name: 'Bicarbonate' },
    'BaseExcess': { min: -20, max: 20, unit: 'mmol/L', name: 'Base Excess' },

    // Lab - Electrolytes
    'Calcium': { min: 5, max: 15, unit: 'mg/dL', name: 'Calcium' },
    'Chloride': { min: 70, max: 130, unit: 'mmol/L', name: 'Chloride' },
    'Potassium': { min: 2, max: 8, unit: 'mmol/L', name: 'Potassium' },
    'Magnesium': { min: 0.5, max: 5, unit: 'mg/dL', name: 'Magnesium' },

    // Lab - Liver
    'AST': { min: 5, max: 5000, unit: 'U/L', name: 'AST' },
    'ALT': { min: 5, max: 5000, unit: 'U/L', name: 'ALT' },
    'ALP': { min: 20, max: 1000, unit: 'U/L', name: 'Alkaline Phosphatase' },

    // Lab - Urine
    'Urine_output': { min: 0, max: 500, unit: 'mL/hr', name: 'Urine Output' },

    // Biomarkers - Sepsis Markers
    'PCT': { min: 0.01, max: 100, unit: 'ng/mL', name: 'Procalcitonin' },
    'CRP': { min: 0, max: 500, unit: 'mg/L', name: 'C-Reactive Protein' },
    'Presepsin': { min: 100, max: 5000, unit: 'pg/mL', name: 'Presepsin' },
    'IL6': { min: 0, max: 1000, unit: 'pg/mL', name: 'Interleukin-6' },
    'IL1b': { min: 0, max: 200, unit: 'pg/mL', name: 'Interleukin-1β' },

    // Biomarkers - Hematology Extended
    'ESR': { min: 0, max: 150, unit: 'mm/hr', name: 'ESR' },
    'MDW': { min: 15, max: 40, unit: '', name: 'Monocyte Distribution Width' },
    'MPV': { min: 5, max: 15, unit: 'fL', name: 'Mean Platelet Volume' },
    'RDW': { min: 10, max: 25, unit: '%', name: 'Red Cell Distribution Width' },
    'Neutrophils': { min: 0.5, max: 40, unit: 'K/µL', name: 'Neutrophils' },
    'Lymphocytes': { min: 0.2, max: 10, unit: 'K/µL', name: 'Lymphocytes' },

    // Biomarkers - Coagulation
    'DDimer': { min: 0, max: 20, unit: 'μg/mL', name: 'D-Dimer' },
    'PT': { min: 8, max: 50, unit: 'seconds', name: 'Prothrombin Time' },
    'aPTT': { min: 15, max: 100, unit: 'seconds', name: 'aPTT' },
    'INR': { min: 0.5, max: 10, unit: '', name: 'INR' },

    // Biomarkers - Chemistry Extended
    'IonizedCalcium': { min: 0.8, max: 1.5, unit: 'mmol/L', name: 'Ionized Calcium' },
    'Phosphorus': { min: 1, max: 10, unit: 'mg/dL', name: 'Phosphorus' },
    'Albumin': { min: 1.5, max: 6, unit: 'g/dL', name: 'Albumin' },
    'Sodium': { min: 110, max: 170, unit: 'mmol/L', name: 'Sodium' },

    // Calculated Ratios
    'NLR': { min: 0.1, max: 100, unit: '', name: 'Neutrophil-Lymphocyte Ratio' },
    'PLR': { min: 10, max: 1000, unit: '', name: 'Platelet-Lymphocyte Ratio' },
    'AnionGap': { min: 0, max: 40, unit: 'mmol/L', name: 'Anion Gap' }
};

function validateInput(fieldName, value) {
    if (!VALIDATION_RANGES[fieldName]) {
        return { valid: true }; // No validation rule defined
    }

    const range = VALIDATION_RANGES[fieldName];
    const numValue = parseFloat(value);

    if (isNaN(numValue)) {
        return {
            valid: false,
            message: `${range.name} must be a number`
        };
    }

    if (numValue < range.min || numValue > range.max) {
        return {
            valid: false,
            message: `${range.name} must be between ${range.min} and ${range.max}${range.unit ? ' ' + range.unit : ''}`
        };
    }

    return { valid: true };
}

function showFieldError(inputElement, message) {
    // Remove existing error
    clearFieldError(inputElement);

    // Create error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'field-error';
    errorDiv.textContent = message;
    errorDiv.id = `error-${inputElement.id}`;

    // Insert after input
    inputElement.parentNode.insertBefore(errorDiv, inputElement.nextSibling);
}

function clearFieldError(inputElement) {
    const errorDiv = document.getElementById(`error-${inputElement.id}`);
    if (errorDiv) {
        errorDiv.remove();
    }
}

function setupRealTimeValidation() {
    // Attach validation to all number inputs
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('blur', function (e) {
            const fieldName = e.target.id;
            const value = e.target.value;

            // Skip if empty (optional fields)
            if (value === '' || value === null) {
                clearFieldError(e.target);
                e.target.classList.remove('input-error', 'input-valid');
                return;
            }

            // Validate
            const result = validateInput(fieldName, value);

            if (!result.valid) {
                // Show error
                e.target.classList.add('input-error');
                e.target.classList.remove('input-valid');
                showFieldError(e.target, result.message);
            } else {
                // Show success
                e.target.classList.add('input-valid');
                e.target.classList.remove('input-error');
                clearFieldError(e.target);
            }
        });

        // Clear error on focus
        input.addEventListener('focus', function (e) {
            clearFieldError(e.target);
        });
    });
}

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

    // Real-time validation setup
    setupRealTimeValidation();
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
