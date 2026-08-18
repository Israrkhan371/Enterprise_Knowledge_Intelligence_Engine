// EKIE frontend logic. Vanilla JS, no framework, no build step.

// ---------- small helpers ----------

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtDate(iso) {
  if (!iso) return "—";
  try { return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch { return iso; }
}

function truncate(s, n = 220) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n).trim() + "…" : s;
}

let toastTimer;
function toast(msg, kind = "") {
  const box = document.getElementById("toast");
  box.textContent = msg;
  box.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.classList.remove("show"); }, 4200);
}

async function guard(promise, { loadingEl, okMsg } = {}) {
  if (loadingEl) loadingEl.innerHTML = `<div class="empty-state">loading…</div>`;
  try {
    const result = await promise;
    if (okMsg) toast(okMsg, "ok");
    return result;
  } catch (err) {
    const msg = err.status ? `[${err.status}] ${err.message}` : err.message;
    toast(msg, "err");
    if (loadingEl) loadingEl.innerHTML = `<div class="empty-state">Error: ${esc(msg)}</div>`;
    throw err;
  }
}

function requireAdmin() {
  if (!API.adminId()) {
    toast("Set an Admin user ID in the ⚙ Admin tab first.", "err");
    switchTab("settings");
    return false;
  }
  return true;
}

// ---------- tab routing ----------

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
  location.hash = name;
}

document.getElementById("tabnav").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (btn) switchTab(btn.dataset.tab);
});

window.addEventListener("DOMContentLoaded", () => {
  const initial = (location.hash || "#ask").slice(1);
  if (document.getElementById(`panel-${initial}`)) switchTab(initial);
});

// ---------- signal bar ----------

async function refreshSignalBar() {
  const statusBox = document.getElementById("apiStatus");
  try {
    await API.health();
    statusBox.className = "signal-status ok";
    statusBox.querySelector(".status-text").textContent = "online";
  } catch {
    statusBox.className = "signal-status err";
    statusBox.querySelector(".status-text").textContent = "unreachable";
    return;
  }

  try {
    const docs = await API.adminListDocuments({ limit: 1 });
    document.querySelector('[data-signal="docs"]').textContent = docs.total ?? "—";
  } catch { /* admin id may not be set yet; leave dash */ }

  try {
    const usage = await API.adminUsageAnalytics();
    document.querySelector('[data-signal="queries"]').textContent = usage.total_queries ?? "—";
    document.querySelector('[data-signal="flagged"]').textContent = usage.flagged_for_review_count ?? "—";
    document.querySelector('[data-signal="verified"]').textContent = usage.verified_count ?? "—";
  } catch { /* same */ }
}

refreshSignalBar();
setInterval(refreshSignalBar, 30000);

// ==================================================================
// ASK
// ==================================================================

document.getElementById("askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("askInput");
  const query = input.value.trim();
  if (!query) return;

  const thread = document.getElementById("askThread");
  const entry = el(`
    <div class="ask-entry">
      <div class="ask-q">${esc(query)}</div>
      <div class="ask-a"><div class="empty-state">thinking…</div></div>
    </div>
  `);
  thread.prepend(entry);
  input.value = "";

  try {
    const result = await API.ask(query);
    const body = entry.querySelector(".ask-a");
    const sourcesHtml = (result.sources || []).map((s) => `
      <div class="source-chip">
        <span class="idx">[${s.index}]</span>doc ${esc(s.document_id)}
        <span class="snippet">${esc(truncate(s.text, 200))}</span>
      </div>
    `).join("") || `<div class="empty-state">No sources retrieved.</div>`;

    const verified = result.citation_check && result.citation_check.verified;
    const flagCount = (result.citation_check && result.citation_check.flags || []).length;

    body.innerHTML = `
      <div class="ask-a-text">${esc(result.answer)}</div>
      <div class="verify-badge ${verified ? "ok" : "warn"}">
        ${verified ? "✓ citations verified" : `⚠ ${flagCount} citation flag(s)`}
      </div>
      <div class="ask-sources"><h3>Sources</h3>${sourcesHtml}</div>
      <div class="ask-feedback">
        <button class="btn btn-sm btn-ghost" data-fb="true">👍 helpful</button>
        <button class="btn btn-sm btn-ghost" data-fb="false">👎 not helpful</button>
      </div>
    `;
    body.querySelectorAll("[data-fb]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await guard(API.askFeedback(result.usage_log_id, btn.dataset.fb === "true"), { okMsg: "Feedback recorded" });
        body.querySelector(".ask-feedback").innerHTML = `<span class="badge">feedback recorded</span>`;
      });
    });
  } catch (err) {
    entry.querySelector(".ask-a").innerHTML = `<div class="empty-state">Error: ${esc(err.message)}</div>`;
  }
});

// ==================================================================
// SEARCH
// ==================================================================

const searchModeSelect = document.getElementById("searchMode");
searchModeSelect.addEventListener("change", () => {
  document.getElementById("metadataFilters").classList.toggle("hidden", searchModeSelect.value !== "metadata");
});

document.getElementById("searchForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = document.getElementById("searchInput").value.trim();
  const mode = searchModeSelect.value;
  const resultsBox = document.getElementById("searchResults");
  const metaBox = document.getElementById("searchMeta");
  metaBox.textContent = "";

  try {
    let results, extra = "";
    if (mode === "semantic") results = await guard(API.searchSemantic(q), { loadingEl: resultsBox });
    else if (mode === "keyword") results = await guard(API.searchKeyword(q), { loadingEl: resultsBox });
    else if (mode === "hybrid") results = await guard(API.searchHybrid(q), { loadingEl: resultsBox });
    else if (mode === "context-aware") {
      const r = await guard(API.searchContextAware(q, []), { loadingEl: resultsBox });
      results = r.results;
      extra = `rewritten query → "${esc(r.rewritten_query)}"`;
    } else if (mode === "metadata") {
      results = await guard(API.searchMetadata({
        category: document.getElementById("metaCategory").value.trim() || undefined,
        source_type: document.getElementById("metaSourceType").value.trim() || undefined,
        status: document.getElementById("metaStatus").value.trim() || undefined,
      }), { loadingEl: resultsBox });
    }

    metaBox.textContent = `${results.length} result(s)${extra ? " — " + extra : ""}`;
    if (!results.length) {
      resultsBox.innerHTML = `<div class="empty-state">No results.</div>`;
      return;
    }
    resultsBox.innerHTML = "";
    results.forEach((r) => {
      const metaParts = [];
      if (r.rank !== undefined) metaParts.push(`<b>rank</b> ${Number(r.rank).toFixed(4)}`);
      if (r.distance !== undefined) metaParts.push(`<b>distance</b> ${Number(r.distance).toFixed(4)}`);
      if (r.fused_score !== undefined) metaParts.push(`<b>fused</b> ${Number(r.fused_score).toFixed(4)}`);
      if (r.rerank_score !== undefined) metaParts.push(`<b>rerank</b> ${Number(r.rerank_score).toFixed(3)}`);
      if (r.document_id) metaParts.push(`doc ${esc(r.document_id)}`);
      resultsBox.appendChild(el(`
        <div class="result-row">
          <div class="meta">${metaParts.join(" · ")}</div>
          <div class="text">${esc(truncate(r.text || r.title || JSON.stringify(r), 400))}</div>
        </div>
      `));
    });
  } catch { /* guard() already surfaced it */ }
});

// ==================================================================
// GRAPH
// ==================================================================

document.getElementById("loadTechMap").addEventListener("click", async () => {
  const out = document.getElementById("techMapOut");
  const label = document.getElementById("techLabelInput").value.trim() || "TECH";
  const map = await guard(API.technologyMap(label), { loadingEl: out });
  const groups = Object.entries(map || {});
  if (!groups.length) { out.innerHTML = `<div class="empty-state">No relationships found.</div>`; return; }
  out.innerHTML = groups.map(([ecosystem, edges]) => `
    <div style="margin-bottom:0.9rem">
      <div class="badge cyan" style="margin-bottom:0.4rem">${esc(ecosystem)}</div>
      ${edges.map((ed) => `
        <div class="result-row" style="margin-bottom:0.4rem">
          <div class="meta"><b>${esc(ed.source)}</b> —${esc(ed.relation)}→ <b>${esc(ed.target)}</b> · confidence ${ed.confidence}</div>
          <div class="text">${esc(ed.reason)}</div>
        </div>
      `).join("")}
    </div>
  `).join("");
});

document.getElementById("loadSkills").addEventListener("click", async () => {
  const out = document.getElementById("skillsOut");
  const skill = document.getElementById("skillInput").value.trim() || undefined;
  const deps = await guard(API.skillDependencies(skill), { loadingEl: out });
  if (!deps.length) { out.innerHTML = `<div class="empty-state">No dependency chain found.</div>`; return; }
  out.innerHTML = deps.map((d) => `
    <div class="result-row" style="margin-bottom:0.4rem">
      <div class="meta"><b>${esc(d.source)}</b> —${esc(d.relation)}→ <b>${esc(d.target)}</b> · confidence ${d.confidence}</div>
      <div class="text">${esc(d.reason)}</div>
    </div>
  `).join("");
});

document.getElementById("loadExplain").addEventListener("click", async () => {
  const out = document.getElementById("explainOut");
  const source = document.getElementById("relSource").value.trim();
  const target = document.getElementById("relTarget").value.trim();
  if (!source || !target) { toast("Enter both a source and target entity", "err"); return; }
  const res = await guard(API.explainRelationship(source, target), { loadingEl: out });
  out.innerHTML = `<pre class="code-block">${esc(JSON.stringify(res, null, 2))}</pre>`;
});

document.getElementById("loadLearning").addEventListener("click", async () => {
  const out = document.getElementById("learningOut");
  const raw = document.getElementById("learnHistory").value.trim();
  const history = raw ? raw.split(",").map((s) => s.trim()).filter(Boolean) : [];
  const recs = await guard(API.learningRecommendations(history), { loadingEl: out });
  if (!recs.length) { out.innerHTML = `<div class="empty-state">No recommendations.</div>`; return; }
  out.innerHTML = recs.map((r) => `
    <div class="result-row" style="margin-bottom:0.4rem">
      <div class="text"><b>${esc(r.course)}</b> — related to ${esc(r.related_entity)}</div>
    </div>
  `).join("");
});

// ==================================================================
// DOCUMENTS (admin)
// ==================================================================

document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!requireAdmin()) return;
  const fd = new FormData();
  fd.append("file", document.getElementById("uploadFile").files[0]);
  fd.append("source_type", document.getElementById("uploadSourceType").value.trim());
  const title = document.getElementById("uploadTitle").value.trim();
  const catId = document.getElementById("uploadCategoryId").value.trim();
  if (title) fd.append("title", title);
  if (catId) fd.append("category_id", catId);

  const out = document.getElementById("uploadResult");
  const res = await guard(API.adminUploadDocument(fd), { loadingEl: out, okMsg: "Document ingested" });
  out.innerHTML = `Ingested <b>${esc(res.title)}</b> — ${res.chunk_count} chunk(s) — <span class="badge">${esc(res.status)}</span>`;
  loadDocuments();
});

document.getElementById("githubForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!requireAdmin()) return;
  const fd = new FormData();
  fd.append("repo_url", document.getElementById("ghUrl").value.trim());
  const token = document.getElementById("ghToken").value.trim();
  if (token) fd.append("github_token", token);

  const out = document.getElementById("githubResult");
  const res = await guard(API.adminIngestGithub(fd), { loadingEl: out, okMsg: "Repository ingested" });
  out.innerHTML = `Ingested ${res.file_count} file(s) from the repository.`;
  loadDocuments();
});

// IDs the user has checked for bulk approve/reject. Only ever holds IDs of
// documents that are currently "pending" — approved/rejected docs don't get
// a checkbox at all, since bulk-deciding them again is meaningless. Cleared
// on every reload since a fresh render means fresh (unchecked) rows.
const selectedDocIds = new Set();

function updateBulkBar() {
  const count = selectedDocIds.size;
  document.getElementById("bulkSelectedCount").textContent = `${count} selected`;
  document.getElementById("bulkApproveBtn").disabled = count === 0;
  document.getElementById("bulkRejectBtn").disabled = count === 0;
}

async function loadDocuments() {
  if (!requireAdmin()) return;
  const box = document.getElementById("docsTable");
  const status = document.getElementById("docStatusFilter").value;
  const source_type = document.getElementById("docSourceTypeFilter").value.trim();
  const data = await guard(API.adminListDocuments({ status, source_type, limit: 100 }), { loadingEl: box });

  selectedDocIds.clear();
  const selectAllCb = document.getElementById("selectAllPending");
  selectAllCb.checked = false;

  if (!data.documents.length) {
    box.innerHTML = `<div class="empty-state">No documents match.</div>`;
    selectAllCb.disabled = true;
    updateBulkBar();
    return;
  }

  const anyPending = data.documents.some((d) => d.status === "pending");
  selectAllCb.disabled = !anyPending;

  box.innerHTML = `
    <table>
      <thead><tr><th></th><th>Title</th><th>Source</th><th>Status</th><th>Uploaded by</th><th>Updated</th></tr></thead>
      <tbody>
        ${data.documents.map((d) => `
          <tr class="row-clickable" data-id="${esc(d.id)}">
            <td class="checkbox-cell">${d.status === "pending" ? `<input type="checkbox" class="doc-select" data-id="${esc(d.id)}">` : ""}</td>
            <td>${esc(d.title)}</td>
            <td><span class="badge">${esc(d.source_type)}</span></td>
            <td><span class="badge ${d.status === "approved" ? "green" : d.status === "rejected" ? "red" : "amber"}">${esc(d.status)}</span></td>
            <td>${esc(d.uploaded_by || "—")}</td>
            <td>${fmtDate(d.updated_at)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
    <div class="hint">${data.total} total</div>
  `;
  box.querySelectorAll("tr[data-id]").forEach((row) => {
    row.addEventListener("click", () => openDocumentDetail(row.dataset.id));
  });
  box.querySelectorAll(".doc-select").forEach((cb) => {
    // Stop the click reaching the <tr> listener above, or checking a box
    // would also pop the detail drawer open every time.
    cb.addEventListener("click", (e) => e.stopPropagation());
    cb.addEventListener("change", (e) => {
      const id = e.target.dataset.id;
      if (e.target.checked) selectedDocIds.add(id); else selectedDocIds.delete(id);
      const boxes = [...box.querySelectorAll(".doc-select")];
      selectAllCb.checked = boxes.length > 0 && boxes.every((b) => b.checked);
      updateBulkBar();
    });
  });
  updateBulkBar();
}
document.getElementById("reloadDocs").addEventListener("click", loadDocuments);

document.getElementById("selectAllPending").addEventListener("change", (e) => {
  const box = document.getElementById("docsTable");
  box.querySelectorAll(".doc-select").forEach((cb) => {
    cb.checked = e.target.checked;
    if (e.target.checked) selectedDocIds.add(cb.dataset.id); else selectedDocIds.delete(cb.dataset.id);
  });
  updateBulkBar();
});

async function bulkDecideDocuments(decision) {
  if (!requireAdmin()) return;
  const ids = [...selectedDocIds];
  if (!ids.length) return;

  const approveBtn = document.getElementById("bulkApproveBtn");
  const rejectBtn = document.getElementById("bulkRejectBtn");
  approveBtn.disabled = true;
  rejectBtn.disabled = true;

  // Sequential, not Promise.all: keeps this from firing dozens of concurrent
  // writes at once, and lets us report exactly how many of N succeeded if
  // one fails partway through instead of an all-or-nothing result.
  let okCount = 0;
  const failed = [];
  for (const id of ids) {
    try {
      await API.adminApproveDocument(id, decision);
      okCount++;
    } catch (err) {
      failed.push(id);
    }
  }

  if (failed.length === 0) {
    toast(`${okCount} document(s) ${decision}`, "ok");
  } else {
    toast(`${okCount} succeeded, ${failed.length} failed — see console`, "err");
    console.error(`bulk ${decision} failed for document IDs:`, failed);
  }

  loadDocuments();
}
document.getElementById("bulkApproveBtn").addEventListener("click", () => bulkDecideDocuments("approved"));
document.getElementById("bulkRejectBtn").addEventListener("click", () => bulkDecideDocuments("rejected"));

async function openDocumentDetail(id) {
  const drawer = document.getElementById("docDetail");
  drawer.classList.remove("hidden");
  drawer.innerHTML = `<div class="empty-state">loading…</div>`;
  const doc = await guard(API.adminGetDocument(id), { loadingEl: drawer });

  drawer.innerHTML = `
    <button class="btn btn-sm btn-ghost drawer-close" id="closeDocDrawer">close ✕</button>
    <h2>${esc(doc.title)}</h2>
    <dl class="kv">
      <dt>id</dt><dd>${esc(doc.id)}</dd>
      <dt>source_type</dt><dd>${esc(doc.source_type)}</dd>
      <dt>status</dt><dd><span class="badge ${doc.status === "approved" ? "green" : doc.status === "rejected" ? "red" : "amber"}">${esc(doc.status)}</span></dd>
      <dt>version</dt><dd>${doc.version}${doc.supersedes_id ? ` (supersedes ${esc(doc.supersedes_id)})` : ""}</dd>
      <dt>quality_score</dt><dd>${doc.quality_score ?? "not scored yet"}</dd>
      <dt>chunks</dt><dd>${doc.chunk_count}</dd>
      <dt>uploaded_by</dt><dd>${esc(doc.uploaded_by || "—")}</dd>
      <dt>created</dt><dd>${fmtDate(doc.created_at)}</dd>
    </dl>
    <div class="drawer-actions">
      <button class="btn btn-sm" id="approveDoc">Approve</button>
      <button class="btn btn-sm btn-danger" id="rejectDoc">Reject</button>
      <button class="btn btn-sm btn-ghost" id="scoreDoc">Score quality</button>
      <button class="btn btn-sm btn-ghost" id="suggestDoc">Suggest updates</button>
      <button class="btn btn-sm btn-ghost" id="versionCandidatesDoc">Version candidates</button>
    </div>
    <div id="docActionOut" class="inline-result"></div>
  `;

  document.getElementById("closeDocDrawer").addEventListener("click", () => drawer.classList.add("hidden"));
  const actionOut = document.getElementById("docActionOut");

  document.getElementById("approveDoc").addEventListener("click", async () => {
    await guard(API.adminApproveDocument(id, "approved"), { okMsg: "Approved" });
    openDocumentDetail(id); loadDocuments();
  });
  document.getElementById("rejectDoc").addEventListener("click", async () => {
    await guard(API.adminApproveDocument(id, "rejected"), { okMsg: "Rejected" });
    openDocumentDetail(id); loadDocuments();
  });
  document.getElementById("scoreDoc").addEventListener("click", async () => {
    const r = await guard(API.adminScoreQuality(id), { loadingEl: actionOut });
    actionOut.innerHTML = `overall <b>${r.overall_score}</b> — completeness ${r.completeness_score}, freshness ${r.freshness_score}, originality ${r.originality_score}`;
  });
  document.getElementById("suggestDoc").addEventListener("click", async () => {
    const r = await guard(API.adminSuggestUpdates(id), { loadingEl: actionOut });
    actionOut.innerHTML = `<div style="white-space:pre-wrap">${esc(r.suggested_updates)}</div>` +
      (r.related_documents?.length ? `<div class="hint">based on: ${r.related_documents.map((d) => esc(d.title)).join(", ")}</div>` : "");
  });
  document.getElementById("versionCandidatesDoc").addEventListener("click", async () => {
    const candidates = await guard(API.adminVersionCandidates(id), { loadingEl: actionOut });
    if (!candidates.length) { actionOut.innerHTML = `<div class="empty-state">No version candidates.</div>`; return; }
    actionOut.innerHTML = candidates.map((c) => `
      <div class="result-row" style="margin-bottom:0.4rem">
        <div class="meta"><b>${esc(c.title)}</b> · similarity ${c.similarity.toFixed(3)} · v${c.version}</div>
        <button class="btn btn-sm btn-ghost" data-supersede="${esc(c.document_id)}">This document supersedes it</button>
      </div>
    `).join("");
    actionOut.querySelectorAll("[data-supersede]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await guard(API.adminLinkVersion(id, btn.dataset.supersede), { okMsg: "Version linked" });
        openDocumentDetail(id);
      });
    });
  });
}

// ==================================================================
// ANSWERS (admin)
// ==================================================================

async function loadAnswers() {
  if (!requireAdmin()) return;
  const box = document.getElementById("answersTable");
  const flagged = document.getElementById("ansFlaggedFilter").value;
  const reviewed = document.getElementById("ansReviewedFilter").value;
  const search = document.getElementById("ansSearchFilter").value.trim();
  const data = await guard(API.adminListAnswers({
    flagged_for_review: flagged || undefined,
    reviewed: reviewed || undefined,
    search: search || undefined,
    limit: 100,
  }), { loadingEl: box });

  if (!data.answers.length) { box.innerHTML = `<div class="empty-state">No answers match.</div>`; return; }
  box.innerHTML = `
    <table>
      <thead><tr><th>Query</th><th>Score</th><th>Flagged</th><th>Reviewed</th><th>Asked</th></tr></thead>
      <tbody>
        ${data.answers.map((a) => `
          <tr class="row-clickable" data-id="${esc(a.id)}">
            <td>${esc(truncate(a.query, 80))}</td>
            <td>${a.retrieval_score ?? "—"}</td>
            <td>${a.flagged_for_review ? '<span class="badge red">flagged</span>' : '<span class="badge">—</span>'}</td>
            <td>${a.reviewed ? '<span class="badge green">reviewed</span>' : '<span class="badge amber">pending</span>'}</td>
            <td>${fmtDate(a.created_at)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
    <div class="hint">${data.total} total</div>
  `;
  box.querySelectorAll("tr[data-id]").forEach((row) => row.addEventListener("click", () => openAnswerDetail(row.dataset.id)));
}
document.getElementById("reloadAnswers").addEventListener("click", loadAnswers);

async function openAnswerDetail(id) {
  const drawer = document.getElementById("ansDetail");
  drawer.classList.remove("hidden");
  drawer.innerHTML = `<div class="empty-state">loading…</div>`;
  const [detail, history] = await Promise.all([
    guard(API.adminGetAnswer(id), { loadingEl: drawer }),
    API.adminAnswerReviewHistory(id).catch(() => []),
  ]);

  drawer.innerHTML = `
    <button class="btn btn-sm btn-ghost drawer-close" id="closeAnsDrawer">close ✕</button>
    <h2>${esc(detail.query)}</h2>
    <div class="ask-a-text" style="margin-bottom:1rem">${esc(detail.answer || "(no answer text)")}</div>
    <dl class="kv">
      <dt>retrieval_score</dt><dd>${detail.retrieval_score ?? "—"}</dd>
      <dt>citation_verified</dt><dd>${detail.citation_verified === null ? "—" : detail.citation_verified}</dd>
      <dt>was_helpful</dt><dd>${detail.was_helpful === null ? "no feedback" : detail.was_helpful}</dd>
      <dt>flagged_for_review</dt><dd>${detail.flagged_for_review}</dd>
      <dt>reviewed</dt><dd>${detail.reviewed}</dd>
    </dl>
    ${detail.citation_flags?.length ? `<div class="hint">Flags: ${detail.citation_flags.map((f) => esc(f.issue || JSON.stringify(f))).join("; ")}</div>` : ""}
    <div class="drawer-actions">
      <button class="btn btn-sm" data-decision="approved">Approve</button>
      <button class="btn btn-sm btn-danger" data-decision="flagged">Flag</button>
      <button class="btn btn-sm btn-ghost" data-decision="dismissed">Dismiss</button>
    </div>
    <div id="ansActionOut" class="inline-result"></div>
    <h3 style="font-family:var(--mono);font-size:0.78rem;color:var(--text-faint);margin-top:1.2rem">Review history</h3>
    ${history.length ? history.map((h) => `
      <div class="result-row" style="margin-bottom:0.4rem">
        <div class="meta"><b>${esc(h.decision)}</b> by ${esc(h.reviewer || "unknown")} · ${fmtDate(h.created_at)}</div>
        ${h.comment ? `<div class="text">${esc(h.comment)}</div>` : ""}
      </div>
    `).join("") : `<div class="empty-state">No review decisions yet.</div>`}
  `;
  document.getElementById("closeAnsDrawer").addEventListener("click", () => drawer.classList.add("hidden"));
  drawer.querySelectorAll("[data-decision]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await guard(API.adminReviewAnswer(id, btn.dataset.decision), { okMsg: `Marked ${btn.dataset.decision}` });
      openAnswerDetail(id); loadAnswers();
    });
  });
}

// ==================================================================
// QUALITY
// ==================================================================

document.getElementById("runScoreAll").addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const out = document.getElementById("scoresOut");
  const scores = await guard(API.adminQualityScoreAll(), { loadingEl: out, okMsg: "Scored all documents" });
  renderScoreTable(out, scores);
});
function renderScoreTable(out, scores) {
  if (!scores.length) { out.innerHTML = `<div class="empty-state">No documents to score.</div>`; return; }
  out.innerHTML = `
    <table>
      <thead><tr><th>Title</th><th>Overall</th><th>Completeness</th><th>Freshness</th><th>Originality</th><th>Words</th></tr></thead>
      <tbody>${scores.map((s) => `
        <tr>
          <td>${esc(s.title)}</td>
          <td><b>${s.overall_score}</b></td>
          <td>${s.completeness_score}</td>
          <td>${s.freshness_score}</td>
          <td>${s.originality_score}</td>
          <td>${s.word_count}</td>
        </tr>
      `).join("")}</tbody>
    </table>
  `;
}

document.getElementById("loadDuplicates").addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const out = document.getElementById("duplicatesOut");
  const dups = await guard(API.adminQualityDuplicates(), { loadingEl: out });
  if (!dups.length) { out.innerHTML = `<div class="empty-state">No duplicate pairs found.</div>`; return; }
  out.innerHTML = dups.map((d) => `
    <div class="result-row" style="margin-bottom:0.4rem">
      <div class="meta">similarity <b>${d.similarity.toFixed(3)}</b></div>
      <div class="text">${esc(d.document_a)} ↔ ${esc(d.document_b)}</div>
    </div>
  `).join("");
});

document.getElementById("loadOutdated").addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const out = document.getElementById("outdatedOut");
  const staleRaw = document.getElementById("staleDays").value;
  const days = staleRaw === "" ? 180 : parseInt(staleRaw, 10);  const crossCheck = document.getElementById("llmCrossCheck").checked;
  const rows = await guard(API.adminQualityOutdated(days, crossCheck), { loadingEl: out });
  if (!rows.length) { out.innerHTML = `<div class="empty-state">Nothing flagged as outdated.</div>`; return; }
  out.innerHTML = rows.map((r) => `
    <div class="result-row" style="margin-bottom:0.4rem">
      <div class="meta"><b>${esc(r.title)}</b> · last updated ${fmtDate(r.last_updated)}</div>
      ${r.llm_verdict ? `<div class="text">${esc(r.llm_verdict)}</div>` : ""}
    </div>
  `).join("");
});

document.getElementById("loadGaps").addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const out = document.getElementById("gapsOut");
  const rows = await guard(API.adminQualityGaps(), { loadingEl: out });
  if (!rows.length) { out.innerHTML = `<div class="empty-state">No recurring low-scoring queries.</div>`; return; }
  out.innerHTML = `
    <table>
      <thead><tr><th>Query</th><th>Occurrences</th><th>Avg score</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td>${esc(r.query)}</td><td>${r.occurrences}</td><td>${r.avg_score}</td></tr>`).join("")}</tbody>
    </table>
  `;
});

document.getElementById("loadMissing").addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const out = document.getElementById("missingOut");
  const minMentionsRaw = document.getElementById("minMentions").value;
  const minMentions = minMentionsRaw === "" ? 3 : parseInt(minMentionsRaw, 10);  const rows = await guard(API.adminMissingKnowledge(minMentions), { loadingEl: out });
  if (!rows.length) { out.innerHTML = `<div class="empty-state">No missing-knowledge alerts at this threshold.</div>`; return; }
  out.innerHTML = `
    <table>
      <thead><tr><th>Entity</th><th>Label</th><th>Mentioned in</th><th>Documents</th></tr></thead>
      <tbody>${rows.map((r) => `
        <tr>
          <td><b>${esc(r.entity)}</b></td>
          <td><span class="badge cyan">${esc(r.label)}</span></td>
          <td>${r.mentioned_in_document_count} docs</td>
          <td>${esc((r.mentioning_documents || []).slice(0, 3).join(", "))}${r.mentioning_documents?.length > 3 ? "…" : ""}</td>
        </tr>
      `).join("")}</tbody>
    </table>
  `;
});

// ==================================================================
// ANALYTICS
// ==================================================================

async function loadAnalyticsSummary() {
  if (!requireAdmin()) return;
  const out = document.getElementById("analyticsSummary");
  const stats = await guard(API.adminUsageAnalytics(), { loadingEl: out });
  const boxes = [
    ["total_queries", "queries"], ["avg_retrieval_score", "avg score"],
    ["helpful_count", "helpful"], ["unhelpful_count", "not helpful"],
    ["verified_count", "citation-verified"], ["flagged_for_review_count", "flagged"],
    ["reviewed_count", "reviewed"], ["pending_review_count", "pending review"],
  ];
  out.innerHTML = boxes.map(([key, label]) => `
    <div class="stat-box"><div class="num">${stats[key] ?? "—"}</div><div class="lbl">${label}</div></div>
  `).join("");
}

document.getElementById("loadTimeseries").addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const out = document.getElementById("timeseriesChart");
  const daysRaw = document.getElementById("daysInput").value;
  const days = daysRaw === "" ? 14 : parseInt(daysRaw, 10);  const points = await guard(API.adminUsageTimeseries(days), { loadingEl: out });
  if (!points.length) { out.innerHTML = `<div class="empty-state">No queries in this range.</div>`; return; }
  const max = Math.max(...points.map((p) => p.query_count), 1);
  out.innerHTML = points.map((p) => `
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="height:${Math.max(4, (p.query_count / max) * 100)}%" title="${p.query_count} queries, ${p.helpful_count} helpful, ${p.flagged_count} flagged"></div>
      <div class="chart-label">${esc(p.date.slice(5))}</div>
    </div>
  `).join("");
});

document.getElementById("loadTopQueries").addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const out = document.getElementById("topQueriesOut");
  const rows = await guard(API.adminTopQueries(15), { loadingEl: out });
  if (!rows.length) { out.innerHTML = `<div class="empty-state">No queries logged yet.</div>`; return; }
  out.innerHTML = `
    <table>
      <thead><tr><th>Query</th><th>Occurrences</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td>${esc(r.query)}</td><td>${r.occurrences}</td></tr>`).join("")}</tbody>
    </table>
  `;
});

// ==================================================================
// CATEGORIES
// ==================================================================

document.getElementById("categoryForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!requireAdmin()) return;
  const name = document.getElementById("catName").value.trim();
  const description = document.getElementById("catDesc").value.trim();
  await guard(API.adminCreateCategory(name, description), { okMsg: "Category created" });
  e.target.reset();
  loadCategories();
});

async function loadCategories() {
  if (!requireAdmin()) return;
  const out = document.getElementById("categoriesOut");
  const cats = await guard(API.adminListCategories(), { loadingEl: out });
  if (!cats.length) { out.innerHTML = `<div class="empty-state">No categories yet.</div>`; return; }
  out.innerHTML = `
    <table>
      <thead><tr><th>Name</th><th>Description</th><th>ID</th></tr></thead>
      <tbody>${cats.map((c) => `<tr><td><b>${esc(c.name)}</b></td><td>${esc(c.description || "—")}</td><td class="hint">${esc(c.id)}</td></tr>`).join("")}</tbody>
    </table>
  `;
}

// ==================================================================
// SETTINGS
// ==================================================================

const adminIdInput = document.getElementById("adminIdInput");
const apiBaseInput = document.getElementById("apiBaseInput");
adminIdInput.value = API.adminId();
apiBaseInput.value = API.base();

document.getElementById("saveAdminId").addEventListener("click", () => {
  localStorage.setItem("ekie_admin_id", adminIdInput.value.trim());
  document.getElementById("settingsSaved").textContent = "Saved.";
  toast("Admin ID saved", "ok");
  refreshSignalBar();
});
document.getElementById("saveApiBase").addEventListener("click", () => {
  localStorage.setItem("ekie_api_base", apiBaseInput.value.trim() || "/api/v1");
  toast("API base saved", "ok");
  refreshSignalBar();
});

// ---------- lazy-load each panel's data the first time it's opened ----------

const loadedOnce = new Set();
document.getElementById("tabnav").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (!btn || loadedOnce.has(btn.dataset.tab)) return;
  loadedOnce.add(btn.dataset.tab);
  if (btn.dataset.tab === "documents") loadDocuments().catch(() => {});
  if (btn.dataset.tab === "answers") loadAnswers().catch(() => {});
  if (btn.dataset.tab === "analytics") loadAnalyticsSummary().catch(() => {});
  if (btn.dataset.tab === "categories") loadCategories().catch(() => {});
});
