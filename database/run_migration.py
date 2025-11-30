"""
Run Database Migration for Phase 2 Biomarkers
==============================================

Executes schema_extension_phase2.sql via Python
"""

import psycopg2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from auth import DB_CONFIG


def run_migration():
    """Execute schema extension SQL"""
    
    print("=" * 60)
    print("DATABASE MIGRATION - PHASE 2 BIOMARKERS")
    print("=" * 60)
    
    # Read SQL file
    sql_file = os.path.join(os.path.dirname(__file__), 'schema_extension_phase2.sql')
    
    print(f"\n[1/2] Reading SQL from {sql_file}")
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_commands = f.read()
    
    # Connect and execute
    print(f"\n[2/2] Executing migration...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    try:
        # Execute all commands
        cur.execute(sql_commands)
        conn.commit()
        
        print("\n[OK] Migration completed successfully!")
        print("\n" + "=" * 60)
        print("DATABASE STRUCTURE UPDATED")
        print("=" * 60)
        
        # Verify new column
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'hourly_data' 
            AND column_name = 'vital_signs_extended'
        """)
        
        result = cur.fetchone()
        if result:
            print(f"\n[OK] Verified: {result[0]} column added (type: {result[1]})")
        
        # Count reference ranges
        cur.execute("SELECT COUNT(*) FROM biomarker_references")
        count = cur.fetchone()[0]
        print(f"[OK] Biomarker reference ranges: {count} entries")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
        
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    success = run_migration()
    sys.exit(0 if success else 1)
