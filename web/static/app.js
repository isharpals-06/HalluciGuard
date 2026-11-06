(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };
  const fmt = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
  const label = (value) => String(value ?? '').replaceAll('_', ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());

  // The sidebar is expanded by default on wide screens and icon-only on medium screens.
  const sidebar = $('#sidebar');
  const toggle = $('#sidebar-toggle');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => {
      const collapsed = document.body.classList.toggle('sidebar-collapsed');
      toggle.setAttribute('aria-expanded', String(!collapsed));
      toggle.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
      toggle.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    });
  }
  const mobileMenu = $('#mobile-menu');
  const scrim = $('#mobile-scrim');
  const closeMobile = () => { sidebar?.classList.remove('mobile-open'); scrim?.classList.remove('visible'); };
  mobileMenu?.addEventListener('click', () => { sidebar?.classList.add('mobile-open'); scrim?.classList.add('visible'); });
  scrim?.addEventListener('click', closeMobile);
  $$('.side-link').forEach((link) => link.addEventListener('click', closeMobile));

  const page = document.body.dataset.page;
  if (page === 'verify') initVerify();
  if (page === 'evaluation') initEvaluation();
  if (page === 'knowledge') initKnowledge();
  if (page === 'architecture') initArchitecture();

  async function getJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    return response.json();
  }
  async function initVerify() {
    const input = $('#input-text');
    const counter = $('#word-count');
    const threshold = $('#abstain-threshold');
    const thresholdValue = $('#threshold-value');
    const advanced = $('#advanced-settings');
    const advancedButton = $('#advanced-toggle');
    let thresholdTouched = false;
    try {
      const info = await getJson('/api/info');
      const savedThreshold = info.execution_modes?.selected_abstention_threshold;
      if (Number.isFinite(Number(savedThreshold))) {
        threshold.value = String(savedThreshold);
        thresholdValue.textContent = `Saved default (${Number(savedThreshold).toFixed(2)})`;
      }
    } catch { /* The saved API default still applies if runtime metadata is unavailable. */ }
    const updateCount = () => {
      const text = input.value;
      const count = text.trim() ? text.trim().split(/\s+/).length : 0;
      counter.textContent = `${count} ${count === 1 ? 'word' : 'words'} · ${text.length.toLocaleString()} / 4,096 characters`;
    };
    input.addEventListener('input', updateCount);
    advancedButton.addEventListener('click', () => {
      const expanded = advanced.hidden;
      advanced.hidden = !expanded;
      advancedButton.setAttribute('aria-expanded', String(expanded));
      advancedButton.querySelector('span').textContent = expanded ? '−' : '＋';
    });
    threshold.addEventListener('input', () => { thresholdTouched = true; thresholdValue.textContent = Number(threshold.value).toFixed(2); });
    const examples = {
      light: 'The speed of light in a vacuum is exactly 299,792,458 meters per second.',
      sydney: 'The capital city of Australia is Sydney.',
      multi: 'The speed of light in a vacuum is exactly 299,792,458 meters per second. However, the capital city of Australia is Sydney.'
    };
    $$('.example-chip').forEach((chip) => chip.addEventListener('click', () => {
      input.value = examples[chip.dataset.example] || '';
      updateCount(); input.focus();
    }));
    $('#btn-verify').addEventListener('click', runVerification);
    $('#export-results').addEventListener('click', () => {
      if (!window.lastVerificationResult) return;
      const blob = new Blob([JSON.stringify(window.lastVerificationResult, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `halluciguard-${window.lastVerificationResult.analysis_id || 'analysis'}.json`;
      anchor.click(); URL.revokeObjectURL(url);
    });
    async function runVerification() {
      const text = input.value.trim();
      const message = $('#verify-message');
      const button = $('#btn-verify');
      const spinner = $('.button-spinner', button);
      if (!text) { message.textContent = 'Enter text to analyze.'; message.className = 'inline-message error'; input.focus(); return; }
      message.textContent = 'Sending text to the local verification API…'; message.className = 'inline-message';
      button.disabled = true; spinner.hidden = false;
      const body = {text, query: $('#input-query').value.trim(), provider: $('#provider-select').value};
      if (!advanced.hidden && thresholdTouched) body.abstain_threshold = Number(threshold.value);
      try {
        const response = await fetch('/api/verify', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `Verification failed (${response.status})`);
        window.lastVerificationResult = data;
        renderVerification(data);
        message.textContent = `Analysis completed in ${data.processing_time_ms ?? '—'} ms.`;
        $('#results-area').hidden = false;
        $('#results-area').scrollIntoView({behavior: 'smooth', block: 'start'});
      } catch (error) { message.textContent = error.message || 'Verification request failed.'; message.className = 'inline-message error'; }
      finally { button.disabled = false; spinner.hidden = true; }
    }
    function renderVerification(data) {
      const metrics = data.metrics || {};
      const stats = [
        ['OVERALL STATUS', label(data.overall_status)],
        ['CLAIMS', metrics.total_claims ?? (data.claims || []).length],
        ['EVIDENCE COVERAGE', `${fmt(metrics.evidence_coverage * 100, 1)}%`],
        ['ABSTAINED', metrics.abstained_claims ?? 0],
        ['MEAN UNCERTAINTY', fmt(metrics.mean_uncertainty)]
      ];
      const summary = $('#results-summary'); summary.replaceChildren();
      stats.forEach(([name, value]) => { const card = make('div', 'summary-stat'); card.append(make('span', '', name), make('strong', '', value)); summary.append(card); });
      $('#result-run-meta').textContent = `Analysis ${data.analysis_id || '—'} · ${data.execution_modes?.dense_retrieval || 'retrieval mode unavailable'} · ${data.execution_modes?.nli || 'NLI mode unavailable'} · ${data.processing_time_ms ?? '—'} ms`;
      const list = $('#claims-container'); list.replaceChildren();
      (data.claims || []).forEach((claim, index) => {
        const card = make('article', 'claim-card');
        const isAbstained = Boolean(claim.abstained);
        const outcome = claim.claim_status === 'INSUFFICIENT_EVIDENCE' || claim.status === 'UNVERIFIED' ? 'INSUFFICIENT EVIDENCE' : label(claim.status);
        const summaryButton = make('button', 'claim-summary'); summaryButton.type = 'button'; summaryButton.setAttribute('aria-expanded', 'false');
        summaryButton.setAttribute('aria-label', `Claim ${index + 1}: ${claim.claim_text}. ${isAbstained ? 'Abstained' : outcome}. Expand details.`);
        summaryButton.append(make('span', 'claim-number', String(index + 1).padStart(2, '0')), make('span', 'claim-summary-text', claim.claim_text || 'Claim text unavailable'));
        const badges = make('span', 'claim-badges'); badges.append(make('span', `status-badge${isAbstained ? ' status-abstained' : ''}`, isAbstained ? `ABSTAINED · ${outcome}` : outcome)); summaryButton.append(badges, make('span', 'claim-chevron', '⌄'));
        const detail = make('div', 'claim-detail'); detail.hidden = true;
        const left = make('div', 'detail-column');
        left.append(detailBlock('ASSESSMENT', claim.explanation || 'No explanation was returned.'), detailBlock('CLAIM SIGNALS', ''));
        const signalBlock = left.lastChild; const signalGrid = make('div', 'signal-list');
        const signals = claim.signals || {};
        [['Confidence', claim.confidence], ['Uncertainty', claim.uncertainty_score], ['Evidence retrieved', claim.evidence?.length > 0 ? 'Yes' : 'No'], ['Evidence condition', label(claim.evidence_condition)], ['Retrieval relevance', signals.retrieval_confidence], ['Semantic similarity', signals.semantic_similarity], ['NLI entailment', signals.nli_entailment], ['NLI contradiction', signals.nli_contradiction], ['Source reliability', signals.source_reliability], ['Freshness confidence', signals.freshness_confidence]].forEach(([name, value]) => {
          if (value === undefined) return;
          const item = make('div', 'signal-item'); item.append(make('span', '', name), make('strong', '', typeof value === 'number' ? fmt(value) : value)); signalGrid.append(item);
        }); signalBlock.append(signalGrid);
        const right = make('div', 'detail-column');
        right.append(make('div', 'eyebrow', 'RETRIEVED EVIDENCE'));
        const evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
        if (!evidence.length) right.append(make('div', 'empty-state', 'No relevant evidence was returned for this claim. The result does not imply contradiction.'));
        evidence.forEach((item) => {
          const ev = make('article', 'evidence-item'); const head = make('div', 'evidence-meta'); head.append(make('span', '', item.okf_object_id || item.evidence_id || 'Evidence artifact'), make('span', '', `Relevance ${fmt(item.relevance_score)}`));
          ev.append(head, make('p', '', item.text || 'Evidence text unavailable'));
          const provenance = claim.provenance || {}; const meta = [provenance.source_name, provenance.source_type, provenance.freshness_state].filter(Boolean).join(' · ');
          if (meta) ev.append(make('div', 'evidence-meta', meta));
          right.append(ev);
        });
        if (isAbstained || claim.abstention_reason) right.append(make('div', 'review-note', `Human review / abstention: ${claim.abstention_reason || 'The system abstained for this claim.'}`));
        const provenance = claim.provenance || {};
        if (provenance.source_name) right.append(detailBlock('PROVENANCE', `${provenance.source_name} · ${provenance.source_type || 'source type unavailable'} · ${provenance.verified ? 'verified artifact' : 'not marked verified'}`));
        detail.append(left, right); summaryButton.addEventListener('click', () => { const open = detail.hidden; detail.hidden = !open; card.classList.toggle('open', open); summaryButton.setAttribute('aria-expanded', String(open)); });
        card.append(summaryButton, detail); list.append(card);
      });
      if (!(data.claims || []).length) list.append(make('div', 'empty-state', 'No atomic claims were extracted from this input.'));
      function detailBlock(title, text) { const block = make('section', 'detail-block'); block.append(make('h3', '', title)); if (text) block.append(make('p', '', text)); return block; }
    }
  }

  async function initEvaluation() {
    const body = $('#benchmark-table tbody');
    try {
      const data = await getJson('/api/evaluation'); const meta = data.metadata || {}; const metrics = data.test_metrics || {};
      $('#eval-samples').textContent = (data.test_predictions || []).length || Object.values(metrics.per_class || {}).reduce((n, c) => n + (c.support || 0), 0) || '—';
      $('#eval-retrieval').textContent = meta.dense_retrieval_mode || '—'; $('#eval-nli').textContent = meta.nli_mode || '—';
      $('#eval-coverage').textContent = `${fmt((data.selective_metrics?.coverage ?? 1) * 100, 1)}%`;
      $('#eval-run-label').textContent = `Seed ${meta.seed ?? '—'} · ${meta.split_grouping || 'group split'} · ${meta.evidence_mode || '—'} evidence`;
      body.replaceChildren();
      const rows = Object.entries(data.baseline_comparison || {}).filter(([key, value]) => typeof value === 'object' && value && !['selective_policy', 'run_metadata'].includes(key));
      rows.forEach(([key, item]) => {
        const tr = document.createElement('tr'); [label(key), fmt(item.accuracy, 4), fmt(item.macro_f1, 4), fmt(item.macro_precision, 4), fmt(item.macro_recall, 4), fmt(item.expected_calibration_error, 4), fmt(item.brier_score, 4)].forEach((value) => tr.append(make('td', '', value))); body.append(tr);
      });
      const matrixBody = $('#confusion-matrix tbody'); matrixBody.replaceChildren();
      const classNames = ['SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT EVIDENCE'];
      (metrics.confusion_matrix || []).forEach((row, index) => {
        const tr = document.createElement('tr'); tr.append(make('td', '', classNames[index] || `Class ${index + 1}`));
        row.forEach((value) => tr.append(make('td', '', value))); matrixBody.append(tr);
      });
      const policy = data.selective_metrics || {};
      const items = [['Threshold', meta.selected_abstention_threshold], ['Coverage', policy.coverage], ['Selective risk', policy.selective_risk], ['Validation target risk', meta.target_selective_risk]];
      const box = $('#selective-summary'); box.replaceChildren();
      items.forEach(([name, value]) => { const item = make('div', 'selective-item'); item.append(make('span', '', name), make('strong', '', value === undefined ? 'Not recorded' : (name === 'Coverage' || name === 'Selective risk' ? `${fmt(Number(value) * 100, 1)}%` : fmt(value)))); box.append(item); });
      try {
        const experiments = await getJson('/api/experiments');
        const ablations = Object.entries(experiments).filter(([key, value]) => key.toLowerCase().startsWith('ablation_') && value && typeof value === 'object');
        if (ablations.length) {
          const panel = $('#saved-ablation-panel'); panel.hidden = false;
          const ablationBody = $('#ablation-table tbody'); ablationBody.replaceChildren();
          ablations.forEach(([key, item]) => {
            const tr = document.createElement('tr');
            [label(key), fmt(item.accuracy, 4), fmt(item.macro_f1, 4), fmt(item.expected_calibration_error, 4), item.coverage === undefined ? 'Not recorded' : `${fmt(item.coverage * 100, 1)}%`].forEach((value) => tr.append(make('td', '', value)));
            ablationBody.append(tr);
          });
        }
      } catch { /* A separate ablation artifact is optional. */ }
      const note = $('#evaluation-caveat span');
      const testCount = (data.test_predictions || []).length;
      if (testCount) note.textContent = `Current saved run: ${testCount} test examples. Calibration and selective metrics are preliminary at this sample size.`;
    } catch (error) { body.replaceChildren(); const row = document.createElement('tr'); row.append(make('td', '', `Could not load saved run: ${error.message}`)); row.firstChild.colSpan = 7; body.append(row); }
  }

  async function initKnowledge() {
    const container = $('#knowledge-list'); const search = $('#knowledge-search'); const verification = $('#knowledge-filter'); const freshness = $('#freshness-filter');
    let artifacts = [];
    try {
      artifacts = await getJson('/api/knowledge');
      $('#knowledge-count').textContent = `${artifacts.length} LOCAL ARTIFACTS`;
      [...new Set(artifacts.map((x) => x.freshness).filter(Boolean))].sort().forEach((state) => freshness.append(new Option(label(state), state)));
      const render = () => {
        const q = search.value.trim().toLowerCase(); const verified = verification.value; const fresh = freshness.value;
        const filtered = artifacts.filter((item) => {
          const haystack = [item.id, item.statement, item.entity, item.attribute, item.value, item.source_name, item.source_type].join(' ').toLowerCase();
          return (!q || haystack.includes(q)) && (verified === 'all' || (verified === 'verified') === Boolean(item.verified)) && (fresh === 'all' || item.freshness === fresh);
        });
        container.replaceChildren(); $('#knowledge-empty').hidden = filtered.length > 0;
        filtered.forEach((item) => {
          const card = make('article', 'knowledge-card'); const head = make('div', 'knowledge-card-head'); head.append(make('span', 'knowledge-id', item.id), make('span', 'knowledge-verified', item.verified ? 'VERIFIED RECORD' : 'UNVERIFIED RECORD'));
          card.append(head, make('h3', '', item.entity || item.source_name || 'Knowledge artifact'), make('p', '', item.statement || 'No statement stored.'));
          const meta = make('div', 'knowledge-meta'); [item.source_name, item.source_type, `Freshness: ${item.freshness || 'unknown'}`, `Reliability: ${fmt(item.reliability, 2)}`].filter(Boolean).forEach((part) => meta.append(make('span', '', part)));
          card.append(meta); container.append(card);
        });
        if (!artifacts.length) container.append(make('div', 'empty-state', 'The local repository contains no artifacts.'));
      };
      [search, verification, freshness].forEach((control) => control.addEventListener('input', render));
      [verification, freshness].forEach((control) => control.addEventListener('change', render)); render();
    } catch (error) { container.replaceChildren(make('div', 'empty-state', `Could not load local artifacts: ${error.message}`)); }
  }

  async function initArchitecture() {
    const list = $('#architecture-modes');
    try {
      const info = await getJson('/api/info'); const modes = info.execution_modes || {};
      const rows = [['Retrieval', modes.retrieval_mode], ['NLI', modes.nli_mode], ['Evidence source', modes.evidence_mode], ['Top K', modes.top_k], ['Minimum relevance', modes.min_relevance_threshold], ['Device', info.cuda_available ? 'CUDA available' : 'CPU']];
      list.replaceChildren(); rows.forEach(([key, value]) => { const row = make('div'); row.append(make('dt', '', key), make('dd', '', value ?? 'Not reported')); list.append(row); });
    } catch (error) { list.replaceChildren(make('div', '', `Could not load configuration: ${error.message}`)); }
  }
})();
