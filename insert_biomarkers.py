"""
Script to insert biomarker sections into index.html
"""

# Read biomarker sections
with open('biomarker_sections.html', 'r', encoding='utf-8') as f:
    biomarker_content = f.read()

# Remove first 3 comment lines from biomarker_sections
lines = biomarker_content.split('\n')
biomarker_insert = '\n'.join(lines[3:])  # Skip first 3 comment lines

# Read index.html
with open('index.html', 'r', encoding='utf-8') as f:
    index_content = f.read()

# Find the insertion point (after TroponinI, before submit button)
# Looking for the closing </div> after TroponinI field
search_text = """                        <div class="form-group">
                            <label>TroponinI</label>
                            <input type="number" id="TroponinI" step="0.01" placeholder="<0.04">
                            <small>Troponin I (ng/mL)</small>
                        </div>
                    </div>

                    <button type="submit" class="btn btn-primary btn-large">
                        🔮 Veri Kaydet ve Tahmin Yap
                    </button>"""

replacement_text = """                        <div class="form-group">
                            <label>TroponinI</label>
                            <input type="number" id="TroponinI" step="0.01" placeholder="<0.04">
                            <small>Troponin I (ng/mL)</small>
                        </div>
                    </div>

                    """ + biomarker_insert + """

                    <button type="submit" class="btn btn-primary btn-large">
                        🔮 Veri Kaydet ve Tahmin Yap
                    </button>"""

# Perform replacement
if search_text in index_content:
    new_content = index_content.replace(search_text, replacement_text)
    
    # Write back
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("[OK] Biomarker sections successfully added to index.html!")
    print(f"[OK] Added {len(biomarker_insert.split(chr(10)))} lines of biomarker HTML")
else:
    print("[ERROR] Could not find insertion point in index.html")
    print("File might have been modified. Please check manually.")
