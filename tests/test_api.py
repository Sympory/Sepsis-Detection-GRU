"""Uçtan uca API testleri: tahmin, güvenilirlik, alarm, taburcu, veri seti, denetim kaydı."""

import pytest

from conftest import NORMAL, SEPTIC

pytestmark = pytest.mark.model


@pytest.fixture()
def ml(model_loaded):
    return model_loaded


def add(client, pid, hour, values):
    return client.post(f'/api/patients/{pid}/hourly-data', json={'hour': hour, 'vital_signs': values})


def test_empty_form_rejected(ml, clients, patient):
    assert add(clients['nurse'], patient, 1, {}).status_code == 400


def test_out_of_range_rejected(ml, clients, patient):
    r = add(clients['nurse'], patient, 1, {'HR': 400})
    assert r.status_code == 400 and r.get_json()['details']


def test_insufficient_data(ml, clients, patient):
    r = add(clients['nurse'], patient, 1, {'HR': 90}).get_json()
    assert r['risk_level'] == 'Yetersiz Veri'
    assert r['reliability']['sufficient_data'] is False


def test_septic_scores_higher_and_raises_alarm(ml, clients, patient):
    nurse = clients['nurse']
    r_norm = add(nurse, patient, 1, NORMAL).get_json()
    pid2 = nurse.post('/api/patients', json={'patient_id': 'T2', 'name': 'Septik', 'age': 70}).get_json()['patient_id']
    r_sep = add(nurse, pid2, 1, SEPTIC).get_json()
    assert r_sep['prediction'] > r_norm['prediction']
    assert r_sep['alarm_raised']
    assert 0 <= r_sep['reliability']['score'] <= 100


def test_explanation_present_for_hybrid(ml, clients, patient):
    import config
    if config.FEATURE_SET != 'v4':
        pytest.skip('Açıklama yalnızca v4 modelinde')
    r = add(clients['nurse'], patient, 1, SEPTIC).get_json()
    ex = r['reliability']['explanation']
    assert ex['factors_up'], 'Septik hastada riski artıran faktör olmalı'
    assert all(f['points'] > 0 for f in ex['factors_up'])


def test_skipped_hours_work(ml, clients, patient):
    nurse = clients['nurse']
    assert add(nurse, patient, 2, NORMAL).status_code == 201
    assert add(nurse, patient, 9, NORMAL).status_code == 201


def test_overwrite_keeps_old_values_in_audit(ml, clients, patient, fresh_db):
    nurse = clients['nurse']
    add(nurse, patient, 1, NORMAL)
    r = add(nurse, patient, 1, dict(NORMAL, HR=120)).get_json()
    assert r['overwritten']
    hist = clients['doctor'].get(f'/api/patients/{patient}/history').get_json()['history']
    upd = [h for h in hist if h['action'] == 'observation.update'][0]
    assert upd['details']['old_values']['HR'] == 78 and upd['details']['values']['HR'] == 120


def test_alarm_ack_requires_note(ml, clients, patient):
    add(clients['nurse'], patient, 1, SEPTIC)
    alarm = clients['doctor'].get('/api/alarms').get_json()['alarms'][0]
    assert clients['doctor'].post(f"/api/alarms/{alarm['id']}/acknowledge", json={'note': ''}).status_code == 400
    assert clients['doctor'].post(f"/api/alarms/{alarm['id']}/acknowledge", json={'note': 'Kültür alındı'}).status_code == 200


def test_discharge_requires_outcome_and_locks_entry(ml, clients, patient):
    doc = clients['doctor']
    assert doc.post(f'/api/patients/{patient}/discharge', json={}).status_code == 400
    assert doc.post(f'/api/patients/{patient}/discharge', json={'outcome': {'sepsis': 'yes'}}).status_code == 400
    assert doc.post(f'/api/patients/{patient}/discharge', json={'outcome': {'sepsis': 'no'}}).status_code == 200
    assert add(clients['nurse'], patient, 5, NORMAL).status_code == 400


def test_dataset_export_anonymous_and_labeled(ml, clients, patient):
    nurse, doc = clients['nurse'], clients['doctor']
    for h in (1, 2, 8):
        add(nurse, patient, h, NORMAL)
    doc.post(f'/api/patients/{patient}/discharge', json={'outcome': {'sepsis': 'yes', 'onset_hour': 8}})
    csv = clients['admin'].get('/api/admin/dataset.csv').get_data(as_text=True)
    lines = csv.strip().splitlines()
    header = lines[0].split(',')
    rows = [dict(zip(header, l.split(','))) for l in lines[1:]]
    assert 'Test Hasta' not in csv and 'T1' not in csv
    assert [r['SepsisLabel'] for r in rows] == ['0', '1', '1']    # başlangıçtan 6 saat önce (saat 2) 1 olur
    assert {r['Patient_ID'] for r in rows} == {'L00001'}


def test_unknown_onset_saved_but_not_exported(ml, clients, patient):
    nurse, doc = clients['nurse'], clients['doctor']
    add(nurse, patient, 1, NORMAL)
    r = doc.post(f'/api/patients/{patient}/discharge', json={'outcome': {'sepsis': 'yes', 'onset_unknown': True}})
    assert r.status_code == 200
    assert doc.get(f'/api/patients/{patient}').get_json()['patient']['outcome']['onset_hour'] is None
    csv = clients['admin'].get('/api/admin/dataset.csv').get_data(as_text=True)
    assert len(csv.strip().splitlines()) == 1, 'Başlangıcı bilinmeyen sepsis hastası veri setine girmemeli'
    ds = clients['admin'].get('/api/admin/db-stats').get_json()['dataset']
    assert ds['sepsis_no_onset'] == 1 and ds['exportable_patients'] == 0


def test_clinical_rules_remember_recent_labs(ml, clients, patient):
    """3. saatte girilen yüksek laktat, 4. saatte tekrar girilmese de klinik kurala girmeli."""
    nurse = clients['nurse']
    add(nurse, patient, 3, dict(NORMAL, Lactate=4.0))
    r = add(nurse, patient, 4, NORMAL).get_json()
    lact = [f for f in r['reliability']['red_flags'] if f['feature'] == 'Lactate']
    assert lact and lact[0]['measured_hour'] == 3


def test_clinical_rules_forget_old_labs(ml, clients, patient):
    """24 saatten eski laboratuvar sonucu kurala girmemeli."""
    nurse = clients['nurse']
    add(nurse, patient, 1, dict(NORMAL, Lactate=4.0))
    r = add(nurse, patient, 30, NORMAL).get_json()
    assert not [f for f in r['reliability']['red_flags'] if f['feature'] == 'Lactate']
