"""
Hugging Face Spaces başlangıcı: veritabanını kurar, modeli yükler, örnek hastaları ekler, sunucuyu açar.

Veritabanı /tmp'de durur; Space yeniden başladığında sıfırlanır ve örnek hastalar yeniden eklenir.
Yönetici şifresi Space ayarlarındaki gizli SEPSIS_ADMIN_PASSWORD değişkeninden gelir.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import app as sepsis_app  # noqa: E402
import ml_service as ml  # noqa: E402
from config import HOST, PORT  # noqa: E402
from demo_hastalari import seed  # noqa: E402

sepsis_app.init_app()
if not ml.load_model_and_preprocessing():
    raise SystemExit('Model yüklenemedi')
ml.warm_up()
seed(sepsis_app)
if not os.environ.get('SEPSIS_ADMIN_PASSWORD'):
    print('[WARN] SEPSIS_ADMIN_PASSWORD tanımlı değil: yönetici hesabına giriş yapılamaz.', flush=True)
print(f'[WEB] http://{HOST}:{PORT}', flush=True)
sepsis_app.app.run(host=HOST, port=PORT, threaded=True)
