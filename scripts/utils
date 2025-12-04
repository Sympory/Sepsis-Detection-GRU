# Fix all unicode characters in app.py
import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace common unicode characters
replacements = {
    '✓': '[OK]',
    '✗': '[ERROR]',
    '❌': '[ERROR]',
    '⚠': '[WARNING]',
    '📊': '[INFO]',
    'başlatıldı': 'baslatildi',
    'yüklendi': 'yuklendi',
    'başarıyla': 'basariyla',
    'başarısız': 'basarisiz',
    'oluşturuldu': 'olusturuldu',
    'çalışıyor': 'calisiyor',
    'güncelendi': 'guncellendi',
    'gösteriliyor': 'gosteriliyor',
    'Başarılı': 'Basarili',
    'Başarısız': 'Basarisiz',
    'İşlem': 'Islem',
    'ışık': 'isik',
    'İstatistik': 'Istatistik',
    'özeti': 'ozeti',
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Unicode characters replaced successfully!")
