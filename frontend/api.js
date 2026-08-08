// Thin API client. No framework, no build step — just fetch() wrappers
// matching the exact endpoints in app/api/routes.py and app/admin/routes.py.

const API = (() => {
  function base() {
    return localStorage.getItem("ekie_api_base") || "/api/v1";
  }

  function adminId() {
    return localStorage.getItem("ekie_admin_id") || "";
  }

  function qs(params) {
    const usp = new URLSearchParams();
    Object.entries(params || {}).forEach(([k, v]) => {
      if (v === undefined || v === null || v === "") return;
      if (Array.isArray(v)) v.forEach((item) => usp.append(k, item));
      else usp.append(k, v);
    });
    const s = usp.toString();
    return s ? `?${s}` : "";
  }

  async function request(path, { method = "GET", params, body, isForm = false, admin = false } = {}) {
    const url = `${base()}${path}${qs(params)}`;
    const headers = {};
    if (admin) headers["X-User-Id"] = adminId();
    let payload = body;
    if (body && !isForm) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
    const res = await fetch(url, { method, headers, body: payload });
    let data = null;
    const text = await res.text();
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    if (!res.ok) {
      const detail = (data && (data.detail || data.message)) || res.statusText || "Request failed";
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  return {
    base, adminId,

    // ---- health ----
    health: () => fetch(`${base().replace(/\/api\/v1$/, "")}/health`).then((r) => r.json()),

    // ---- search ----
    searchSemantic: (q, top_k = 10) => request("/search/semantic", { params: { q, top_k } }),
    searchKeyword: (q, top_k = 10) => request("/search/keyword", { params: { q, top_k } }),
    searchHybrid: (q, top_k = 10, category_filter) => request("/search/hybrid", { params: { q, top_k, category_filter } }),
    searchMetadata: (params) => request("/search/metadata", { params }),
    searchContextAware: (q, history, top_k = 10) => request("/search/context-aware", { params: { q, history, top_k } }),

    // ---- ask ----
    ask: (query, history = [], user_id = null) => request("/ask", { method: "POST", body: { query, history, user_id } }),
    askFeedback: (usageLogId, was_helpful) => request(`/ask/${usageLogId}/feedback`, { method: "POST", body: { was_helpful } }),

    // ---- documents (public) ----
    compareDocuments: (a, b) => request("/documents/compare", { method: "POST", body: { document_id_a: a, document_id_b: b } }),
    documentSummary: (id) => request(`/documents/${id}/summary`),
    documentSuggestUpdates: (id) => request(`/documents/${id}/suggest-updates`),

    // ---- graph ----
    technologyMap: (entity_label = "TECH") => request("/graph/technology-map", { params: { entity_label } }),
    skillDependencies: (skill) => request("/graph/skill-dependencies", { params: { skill } }),
    explainRelationship: (source, target) => request("/graph/relationships/explain", { params: { source, target } }),
    learningRecommendations: (user_query_history = []) => request("/graph/learning-recommendations", { params: { user_query_history } }),

    // ---- evaluation ----
    runEvaluation: (k = 10) => request("/evaluation/run", { method: "POST", params: { k } }),

    // ---- admin: documents ----
    adminUploadDocument: (formData) => request("/documents/upload", { method: "POST", body: formData, isForm: true, admin: true }),
    adminIngestGithub: (formData) => request("/documents/ingest-github", { method: "POST", body: formData, isForm: true, admin: true }),
    adminListDocuments: (params) => request("/documents", { params, admin: true }),
    adminGetDocument: (id) => request(`/documents/${id}`, { admin: true }),
    adminApproveDocument: (id, decision, comment = "") => request(`/documents/${id}/approve`, { method: "POST", params: { decision, comment }, admin: true }),
    adminVersionCandidates: (id) => request(`/documents/${id}/version-candidates`, { admin: true }),
    adminLinkVersion: (id, supersedes_id) => request(`/documents/${id}/link-version`, { method: "POST", params: { supersedes_id }, admin: true }),
    adminVersionHistory: (id) => request(`/documents/${id}/version-history`, { admin: true }),
    adminScoreQuality: (id) => request(`/documents/${id}/score-quality`, { method: "POST", admin: true }),
    adminSuggestUpdates: (id) => request(`/documents/${id}/suggest-updates`, { admin: true }),

    // ---- admin: categories ----
    adminListCategories: () => request("/categories", { admin: true }),
    adminCreateCategory: (name, description = "") => request("/categories", { method: "POST", params: { name, description }, admin: true }),

    // ---- admin: answers ----
    adminListAnswers: (params) => request("/answers", { params, admin: true }),
    adminGetAnswer: (id) => request(`/answers/${id}`, { admin: true }),
    adminAnswerReviewHistory: (id) => request(`/answers/${id}/review-history`, { admin: true }),
    adminReviewAnswer: (id, decision, comment = "") => request(`/answers/${id}/review`, { method: "POST", params: { decision, comment }, admin: true }),

    // ---- admin: analytics ----
    adminUsageAnalytics: () => request("/analytics/usage", { admin: true }),
    adminUsageTimeseries: (days = 14) => request("/analytics/usage/timeseries", { params: { days }, admin: true }),
    adminTopQueries: (limit = 10) => request("/analytics/usage/top-queries", { params: { limit }, admin: true }),

    // ---- admin: quality ----
    adminQualityDuplicates: () => request("/quality/duplicates", { admin: true }),
    adminQualityOutdated: (staleness_days = 180, llm_cross_check = false) => request("/quality/outdated", { params: { staleness_days, llm_cross_check }, admin: true }),
    adminQualityGaps: () => request("/quality/gaps", { admin: true }),
    adminQualityScoreAll: () => request("/quality/score-all", { method: "POST", admin: true }),
    adminMissingKnowledge: (min_mentions = 3, entity_label) => request("/quality/missing-knowledge", { params: { min_mentions, entity_label }, admin: true }),
  };
})();
