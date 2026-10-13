"""
Uygulamada biriken veriyi yeniden eğitim için veri setine çevirir.

  py -3.10 scripts/veri_seti_disa_aktar.py                  # data/yerel_veriseti.csv
  py -3.10 scripts/veri_seti_disa_aktar.py --cikti dosya.csv

Yalnızca taburcu edilmiş ve doktorun sonucunu "Evet" ya da "Hayır" olarak işaretlediği
hastalar alınır. Ad ve protokol numarası yazılmaz; her hastaya L00001 gibi bir sıra no verilir.
Format PhysioNet 2019 (data/train.csv) ile aynıdır, dolayısıyla iki veri birleştirilip
scripts/train_v3.py --input ile model yeniden eğitilebilir.

SepsisLabel kuralı (PhysioNet ile aynı): klinik başlangıçtan 6 saat önce 1 olur.
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cikti', default=os.path.join(ROOT, 'data', 'yerel_veriseti.csv'))
    args = parser.parse_args()

    db.init_database()
    conn = db.connect()
    summary = db.dataset_summary(conn)
    header, rows = db.export_dataset_rows(conn)
    db.audit(conn, None, 'dataset.export', details={'rows': len(rows), 'file': os.path.basename(args.cikti), 'via': 'script'})
    conn.commit()
    conn.close()

    os.makedirs(os.path.dirname(args.cikti) or '.', exist_ok=True)
    with open(args.cikti, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Taburcu hastalar: {summary['sepsis_yes']} sepsis, {summary['sepsis_no']} sepsis değil, "
          f"{summary['sepsis_no_onset']} sepsis (saat bilinmiyor), {summary['sepsis_unknown']} belirsiz, {summary['unlabeled']} etiketsiz")
    print(f"Veri setine alınan: {summary['exportable_patients']} hasta, {len(rows)} saatlik satır → {args.cikti}")
    if summary['exportable_patients'] < 100:
        print("Not: Anlamlı bir yeniden eğitim için yüzlerce etiketli hasta gerekir; veri biriktikçe tekrar çalıştırın.")


if __name__ == '__main__':
    main()
