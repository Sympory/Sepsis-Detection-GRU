"""
Hugging Face Space'e yüklenecek klasörü hazırlar: yalnızca uygulamanın çalışması için gereken dosyalar.
Veri seti, veritabanı, eski modeller ve eğitim betikleri kopyalanmaz.

  py -3.10 deploy/hf_space_hazirla.py            → build/hf_space/
  docker build -t sepsisai build/hf_space         (isteğe bağlı: yerelde dene)
"""

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'build', 'hf_space')

APP_FILES = ['app.py', 'config.py', 'db.py', 'security.py', 'ml_service.py', 'features.py', 'features_v4.py',
             'gbm_numpy.py', 'explain.py', 'reliability.py', 'olcek.py',
             'index.html', 'login.html', 'script.js', 'login.js', 'style.css']
MODEL_FILES = ['score_map.json', 'model_card.json', 'mlp.keras', 'trees.json', 'scaler.pkl', 'calibrator.pkl', 'reliability_stats.json',
               'clinical_comparison.json', 'cross_validation.json', 'hastane_arasi_test.json', 'hasta_bazli_test.json']
DEPLOY_FILES = ['Dockerfile', 'requirements.txt', 'space_start.py', 'README.md']


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if os.path.exists(OUT):
        # .git klasörü korunur: Space deposu bu klasörde klonlanmış olabilir
        for name in os.listdir(OUT):
            if name != '.git':
                path = os.path.join(OUT, name)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    os.makedirs(OUT, exist_ok=True)

    for f in APP_FILES:
        copy(os.path.join(ROOT, f), os.path.join(OUT, f))
    for f in MODEL_FILES:
        copy(os.path.join(ROOT, 'models', 'v5', f), os.path.join(OUT, 'models', 'v5', f))
    copy(os.path.join(ROOT, 'scripts', 'demo_hastalari.py'), os.path.join(OUT, 'scripts', 'demo_hastalari.py'))
    for f in DEPLOY_FILES:
        copy(os.path.join(ROOT, 'deploy', 'huggingface', f), os.path.join(OUT, f))

    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) if '.git' not in d for f in fs)
    print(f'Hazır: {OUT} ({size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
