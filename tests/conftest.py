"""
Ortak test düzeni.

Her test oturumu geçici bir veritabanı kullanır; gerçek patients.db'ye dokunulmaz.
Model, yavaş olduğu için oturum başına bir kez yüklenir.

  py -3.10 -m pytest                 # tüm testler
  py -3.10 -m pytest -m "not model"  # modelsiz hızlı testler
"""

import os
import sys
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['SEPSIS_DEMO_ACCOUNTS'] = '1'
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import pytest

DEMO = {'admin': ('yonetici', 'Yonetici123'), 'doctor': ('doktor', 'Doktor123'), 'nurse': ('hemsire', 'Hemsire123')}
NORMAL = {'HR': 78, 'O2Sat': 98, 'Temp': 36.8, 'Resp': 16, 'SBP': 125, 'DBP': 75}
SEPTIC = {'HR': 132, 'O2Sat': 88, 'Temp': 38.9, 'Resp': 29, 'SBP': 82, 'DBP': 48, 'Lactate': 4.5, 'WBC': 17}


def pytest_configure(config):
    config.addinivalue_line('markers', 'model: modeli yükleyen (yavaş) testler')


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    """Her test için boş, geçici veritabanı + demo hesapları."""
    import db
    import app as sepsis_app
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'test.db'))
    sepsis_app.init_app()
    return db


@pytest.fixture(scope='session')
def model_loaded():
    import ml_service as ml
    assert ml.load_model_and_preprocessing(), 'Model yüklenemedi'
    return ml


@pytest.fixture()
def app_module(fresh_db):
    import app as sepsis_app
    return sepsis_app


def login(app_module, role):
    client = app_module.app.test_client()
    username, password = DEMO[role]
    r = client.post('/api/auth/login', json={'username': username, 'password': password})
    assert r.status_code == 200, r.get_json()
    return client


@pytest.fixture()
def clients(app_module):
    return {role: login(app_module, role) for role in DEMO}


@pytest.fixture()
def patient(clients):
    r = clients['nurse'].post('/api/patients', json={'patient_id': 'T1', 'name': 'Test Hasta', 'age': 70, 'gender': 'Male'})
    assert r.status_code == 201
    return r.get_json()['patient_id']
