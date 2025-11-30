"""
Database Initialization Script
===============================

Seeds the database with initial data for testing and demo purposes.
"""

import psycopg2
from psycopg2.extras import execute_values
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from auth import hash_password, DB_CONFIG


def init_database():
    """Initialize database with tables and seed data"""
    
    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("✓ Connected to database")
    except Exception as e:
        print(f"✗ Failed to connect to database: {e}")
        return False
    
    try:
        # Read and execute schema
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        
        print(f"\n[1/3] Executing schema from {schema_path}")
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
            cur.execute(schema_sql)
        conn.commit()
        print("✓ Schema created successfully")
        
        # Seed hospitals
        print("\n[2/3] Seeding hospitals...")
        hospitals = [
            ('Ankara Şehir Hastanesi', 'ASH', 'Ankara', 'Turkey', 'info@ankarashehir.gov.tr', '+90 312 XXX XXXX'),
            ('İstanbul Üniversitesi Hastanesi', 'IUH', 'Istanbul', 'Turkey', 'info@istanbuluni.edu.tr', '+90 212 XXX XXXX'),
            ('Ege Üniversitesi Hastanesi', 'EUH', 'Izmir', 'Turkey', 'info@ege.edu.tr', '+90 232 XXX XXXX'),
            ('Demo Hastane (Test)', 'DEMO', 'Test City', 'Turkey', 'demo@test.com', '+90 555 XXX XXXX'),
        ]
        
        execute_values(cur, """
            INSERT INTO hospitals (name, code, city, country, contact_email, contact_phone)
            VALUES %s
        """, hospitals)
        conn.commit()
        print(f"✓ Inserted {len(hospitals)} hospitals")
        
        # Get hospital IDs
        cur.execute("SELECT id, code FROM hospitals")
        hospital_map = {code: id for id, code in cur.fetchall()}
        
        # Seed users
        print("\n[3/3] Seeding users...")
        
        # Admin password: admin123
        admin_pw = hash_password('admin123')
        
        # Doctor password: doctor123
        doctor_pw = hash_password('doctor123')
        
        # Nurse password: nurse123
        nurse_pw = hash_password('nurse123')
        
        users = [
            # System admin
            ('admin', admin_pw, 'System Administrator', 'admin@system.com', hospital_map['DEMO'], 'admin'),
            
            # Ankara Şehir Hastanesi
            ('ash_admin', admin_pw, 'Mehmet Yılmaz', 'mehmet.yilmaz@ankarashehir.gov.tr', hospital_map['ASH'], 'hospital_admin'),
            ('dr_ayse', doctor_pw, 'Dr. Ayşe Demir', 'ayse.demir@ankarashehir.gov.tr', hospital_map['ASH'], 'doctor'),
            ('hemşire_fatma', nurse_pw, 'Hemşire Fatma Kaya', 'fatma.kaya@ankarashehir.gov.tr', hospital_map['ASH'], 'nurse'),
            
            # İstanbul Üniversitesi
            ('iuh_admin', admin_pw, 'Ali Öztürk', 'ali.ozturk@istanbuluni.edu.tr', hospital_map['IUH'], 'hospital_admin'),
            ('dr_ahmet', doctor_pw, 'Dr. Ahmet Çelik', 'ahmet.celik@istanbuluni.edu.tr', hospital_map['IUH'], 'doctor'),
            ('hemşire_zeynep', nurse_pw, 'Hemşire Zeynep Arslan', 'zeynep.arslan@istanbuluni.edu.tr', hospital_map['IUH'], 'nurse'),
            
            # Ege Üniversitesi
            ('euh_admin', admin_pw, 'Can Yıldız', 'can.yildiz@ege.edu.tr', hospital_map['EUH'], 'hospital_admin'),
            ('dr_elif', doctor_pw, 'Dr. Elif Güneş', 'elif.gunes@ege.edu.tr', hospital_map['EUH'], 'doctor'),
            
            # Demo accounts
            ('demo_doctor', doctor_pw, 'Demo Doctor', 'demo.doctor@test.com', hospital_map['DEMO'], 'doctor'),
            ('demo_nurse', nurse_pw, 'Demo Nurse', 'demo.nurse@test.com', hospital_map['DEMO'], 'nurse'),
        ]
        
        execute_values(cur, """
            INSERT INTO users (username, password_hash, full_name, email, hospital_id, role)
            VALUES %s
        """, users)
        conn.commit()
        print(f"✓ Inserted {len(users)} users")
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATABASE INITIALIZED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📊 Summary:")
        print(f"  • Hospitals: {len(hospitals)}")
        print(f"  • Users: {len(users)}")
        
        print("\n🔑 Demo Login Credentials:")
        print("  ┌─────────────────────────────────────────┐")
        print("  │  Role: System Admin                     │")
        print("  │  Username: admin                        │")
        print("  │  Password: admin123                     │")
        print("  │  Hospital: Demo Hastane (Test)          │")
        print("  ├─────────────────────────────────────────┤")
        print("  │  Role: Doctor                           │")
        print("  │  Username: demo_doctor                  │")
        print("  │  Password: doctor123                    │")
        print("  │  Hospital: Demo Hastane (Test)          │")
        print("  ├─────────────────────────────────────────┤")
        print("  │  Role: Nurse                            │")
        print("  │  Username: demo_nurse                   │")
        print("  │  Password: nurse123                     │")
        print("  │  Hospital: Demo Hastane (Test)          │")
        print("  └─────────────────────────────────────────┘")
        
        print("\n⚠️  IMPORTANT SECURITY NOTES:")
        print("  • Change all default passwords in production!")
        print("  • Configure proper DB_PASSWORD in .env file")
        print("  • Enable SSL/TLS for database connections")
        print("  • Implement proper password policy")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during initialization: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    print("\n⚠️  WARNING: This will DROP all existing tables!")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        success = init_database()
        sys.exit(0 if success else 1)
    else:
        print("Aborted.")
        sys.exit(0)
