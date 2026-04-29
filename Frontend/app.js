// Function to jump to a specific audit page
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}
document.addEventListener('DOMContentLoaded', () => {
    
    // --- Configuration & Elements ---
    const API_BASE = "";
    const scroller = document.querySelector('.scroller');
    const blobs = document.querySelectorAll('.blob');
    const pages = document.querySelectorAll('.page');
    
    // CSV Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('fileInput');
    const tabularResults = document.getElementById('tabular-results');
    const tabularLoader = document.getElementById('tabular-loader');
    const tabularDataView = document.getElementById('tabular-data-view');

    // Text Elements
    const textInput = document.getElementById('text-input');
    const textAuditBtn = document.getElementById('text-audit-btn');
    const textResults = document.getElementById('text-results');
    const textReportContent = document.getElementById('text-report-content');

    // --- 1. Enchanting Parallax Scroll Logic ---
    // Links mouse/scroll position to background blob movement for a deep-space feel
    scroller.addEventListener('scroll', () => {
        const scrollTop = scroller.scrollTop;
        const speedMultiplier = [0.05, 0.08, 0.03]; // Blobs move at different slower speeds than main content
        
        blobs.forEach((blob, index) => {
            const yOffset = scrollTop * speedMultiplier[index];
            blob.style.transform = `translateY(${-yOffset}px)`;
        });
    });

    // --- 2. Scroll-In Animations (Intersection Observer) ---
    const observerOptions = {
        root: scroller,
        threshold: 0.2 // Lowered threshold so it triggers earlier
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
            }
        });
    }, observerOptions);

    pages.forEach(page => observer.observe(page));

    // HOTFIX: Force the first page to reveal immediately on load
    setTimeout(() => {
        const heroPage = document.getElementById('hero');
        if (heroPage) heroPage.classList.add('in-view');
    }, 100);

    // --- 3. Feature: Organizational Audit (CSV Handling) ---
    
    // Handle drop-zone click to trigger native file browser
    dropZone.addEventListener('click', () => fileInput.click());

    // File input selection
    fileInput.addEventListener('change', handleFileSelection);

    // Drag and Drop visual states
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragging');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragging');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelection({ target: { files: files } });
    });

    // Handle the chosen file, show loader, and call backend
    async function handleFileSelection(e) {
        const file = e.target.files[0];
        if (!file || file.type !== 'text/csv') {
            alert('Please provide a valid CSV file.');
            return;
        }

        // UI Transition to Loading State
        dropZone.classList.add('hide-me');
        tabularResults.classList.remove('hide-me');
        tabularLoader.classList.remove('hide-me');
        tabularDataView.classList.add('hide-me');

        // Create form data payload
        const formData = new FormData();
        formData.append("file", file);

        try {
            // CALL BACKEND
            const response = await fetch(`${API_BASE}/upload-data`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Network response failure.');

            const data = await response.json();
            renderTabularResults(data, file.name);

        } catch (error) {
            tabularLoader.innerText = "Error scanning file. Check backend console.";
            console.error(error);
        }
    }

    // Parse JSON results from backend and inject HTML
    // Parse JSON results from Claude's upgraded backend and inject HTML
    function renderTabularResults(data, fileName) {
        tabularLoader.classList.add('hide-me');
        tabularDataView.classList.remove('hide-me');

        // Check the new Claude backend label: "fairness_metrics"
        const alertClass = data.fairness_metrics.bias_detected ? 'alert' : '';

        tabularDataView.innerHTML = `
            <h4 style="color: var(--accent); margin-bottom: 10px;">Audit Report: ${fileName}</h4>
            <div class="gemini-report" style="border-left: 3px solid var(--accent); padding-left: 15px; margin-bottom: 20px;">
                <h5 style="color: #bbb; margin-bottom: 5px;">Ethical Audit :</h5>
                <p style="line-height: 1.5;">${data.gemini_bias_mitigation_report}</p>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div class="stat-card ${alertClass}" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid ${alertClass ? '#ff6b6b' : 'rgba(255,255,255,0.1)'};">
                    <div style="font-size: 2rem; font-weight: bold; color: ${alertClass ? '#ff6b6b' : 'var(--accent)'};">${data.fairness_metrics.disparate_impact_ratio.toFixed(3)}</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Disparate Impact Ratio</div>
                </div>
                <div class="stat-card" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 1.2rem; font-weight: bold; color: #fff;">${data.auto_detected_protected_attribute}</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Detected Demographic</div>
                </div>
            </div>
            <button onclick="window.location.reload()" class="glow-btn" style="margin-top: 20px; width: 100%;">Run New Inspection</button>
        `;
    }

    // --- 4. Feature: Generative AI Audit (Text Handling) ---

    textAuditBtn.addEventListener('click', async () => {
        const content = textInput.value.trim();
        if(!content) return;

        // UI State
        textAuditBtn.innerText = "Analyzing...";
        textAuditBtn.disabled = true;
        textResults.classList.add('hide-me');

        try {
            // CALL BACKEND
            const response = await fetch(`${API_BASE}/audit-text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: content }) // MUST BE 'text'
            });

            if (!response.ok) throw new Error('Text audit failed.');

            const data = await response.json();

            // UI Render for Fact-Checker
            const badgeColor = data.is_correct ? '#00e676' : '#ff6b6b';
            const statusText = data.is_correct ? 'Verified Correct' : 'Correction Required';

            textReportContent.innerHTML = `
                <div style="margin-bottom: 15px;">
                    <span style="background: ${badgeColor}20; color: ${badgeColor}; padding: 5px 10px; border-radius: 5px; font-weight: bold; font-size: 0.8rem; border: 1px solid ${badgeColor};">${statusText}</span>
                </div>
                <p style="margin-bottom: 15px; line-height: 1.5; color: #eee;"><strong>Analysis:</strong> ${data.analysis_report}</p>
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; border-left: 3px solid ${badgeColor};">
                    <strong style="color: var(--accent); font-size: 0.9rem;">Corrected Output:</strong><br>
                    <span style="font-family: monospace; color: #fff; display: inline-block; margin-top: 5px;">${data.corrected_version}</span>
                </div>
            `;
            textResults.classList.remove('hide-me');

        } catch (error) {
            textReportContent.innerText = "Error analyzing text with Gemini API.";
            textResults.classList.remove('hide-me');
            console.error(error);
        } finally {
            textAuditBtn.innerText = "Run Audit";
            textAuditBtn.disabled = false;
        }
    });
});