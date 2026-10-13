// ============================================================================
// BIOMARKER COLLAPSIBLE SECTIONS - Phase 2
// ============================================================================

function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    const header = event.target.closest('.section-toggle');

    if (section) {
        section.classList.toggle('collapsed');
        if (header) {
            header.classList.toggle('collapsed');
        }
    }
}
