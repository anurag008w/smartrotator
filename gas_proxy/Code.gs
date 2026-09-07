/**
 * SmartRotator — Google Apps Script API Gateway (worker-type proxy)
 * ==================================================================
 *
 * Cloudflare Worker ka Google wala version! Same concept:
 *   1. Request aati hai → path/query se provider detect
 *   2. UrlFetchApp se upstream forward (GOOGLE ke IP pool se egress!)
 *   3. Response wapas
 *
 * Egress IP Google ka hai — isliye OpenCode Zen ka Cloudflare IP rate-limit
 * yahan NAHI lagta.
 *
 * FIX (2026-09-07): OpenCode Zen/Go backend ab `x-opencode-session` header
 * require karta hai (2026-09-06 se enforcement). Bina iske:
 *   MissingSessionID: "OpenCode's free tier can only be used in OpenCode"
 * Ab zen provider ke liye official opencode CLI headers inject hote hain:
 *   x-opencode-client / x-opencode-session / x-opencode-project /
 *   x-opencode-request / User-Agent (opencode/<version>)
 * Version dynamic rehta hai (OPENCODE_VERSION script property se override).
 *
 * Deploy:
 *   1. script.google.com → New Project
 *   2. Ye code paste karo (Code.gs)
 *   3. Deploy → New deployment → Web app
 *      - Execute as: Me
 *      - Who has access: Anyone
 *   4. URL milega: https://script.google.com/macros/s/{SCRIPT_ID}/exec
 *   5. SmartRotator me base_url daalo:
 *      https://script.google.com/macros/s/{SCRIPT_ID}/exec?target=zen/v1
 *
 * Routes (query param `target`):
 *   zen        → target=zen/v1          → https://opencode.ai/zen/v1
 *   gemini     → target=gemini/v1beta   → https://generativelanguage.googleapis.com/v1beta
 *   groq       → target=groq/openai/v1  → https://api.groq.com/openai/v1
 *   openrouter → target=openrouter/api/v1 → https://openrouter.ai/api/v1
 *   nvidia     → target=nvidia/v1       → https://integrate.api.nvidia.com/v1
 *
 * NOTE (important limitations):
 *   - Google Apps Script me SSE/streaming support NAHI hai — UrlFetchApp
 *     response ko poora buffer karke deta hai. Isliye stream:true calls
 *     yahan nahi chalenge (non-stream kaam karega). Streaming ke liye
 *     SmartRotator ko DIRECT provider URL use karna chahiye.
 *   - UrlFetchApp timeout ~60 sec (fetchTimeoutSeconds max).
 *   - Cold start ho sakta hai (pehli call 2-5 sec lagti hai).
 *   - 20,000 UrlFetch calls/day (free quota).
 *   - BROWSER CORS: GAS web app custom CORS headers (Allow-Methods /
 *     Allow-Headers) bhej NAHI sakta. Browser se POST + application/json
 *     karne pe preflight OPTIONS fail hota hai → request provider tak nahi
 *     pahunchti. Workarounds:
 *       a) GET + ?target=zen/v1/chat/completions&body=<urlencoded JSON>
 *          (simple request — koi preflight nahi, code iske liye ready hai)
 *       b) POST with Content-Type: text/plain (body me JSON string daalo)
 *     Native/desktop clients (curl, Python, Electron net module) pe CORS
 *     check nahi hota — koi issue nahi.
 */


// ---------------------------------------------------------------------------
// 1) Provider routes — provider ka REAL upstream host
// ---------------------------------------------------------------------------
var PROVIDERS = {
  zen: {
    host: "https://opencode.ai/zen",
    label: "OpenCode Zen",
  },
  gemini: {
    host: "https://generativelanguage.googleapis.com",
    label: "Google Gemini",
  },
  groq: {
    host: "https://api.groq.com",
    label: "Groq",
  },
  openrouter: {
    host: "https://openrouter.ai",
    label: "OpenRouter",
  },
  nvidia: {
    host: "https://integrate.api.nvidia.com",
    label: "NVIDIA NIM",
  },
};


// Browser-ish fingerprints (Google IP se request ab bhi provider ko alag dikhe)
var USER_AGENTS = [
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.2592.87",
];

// OpenCode version — GAS script properties se override kar sakte ho, warna
// default (kal ki latest release). Auto-update: script property set karo:
//   Script Properties → OPENCODE_VERSION = "1.18.29"
function getOpencodeVersion() {
  try {
    var v = PropertiesService.getScriptProperties().getProperty("OPENCODE_VERSION");
    if (v && v.trim()) return v.trim();
  } catch (e) { /* ignore */ }
  return "1.18.29";
}

// OpenCode Zen/Go official CLI headers — bina inke backend `MissingSessionID`
// de deta hai ("free tier can only be used in OpenCode").
function buildOpencodeHeaders() {
  return {
    "x-opencode-client": "cli",
    "x-opencode-session": Utilities.getUuid(),     // per-request unique
    "x-opencode-project": Utilities.getUuid(),
    "x-opencode-request": Utilities.getUuid(),
    "User-Agent": "opencode/" + getOpencodeVersion(),
    "HTTP-Referer": "https://opencode.ai/",
    "X-Title": "opencode",
  };
}


function pick(arr) {
  if (!arr || !arr.length) return "Mozilla/5.0 (compatible; SmartRotator-GAS/1.0)";
  return arr[Math.floor(Math.random() * arr.length)];
}


// ---------------------------------------------------------------------------
// 2.5) fetch with retry — Google ke shared IP pool pe 429/5xx transient hota
//      hai (doosre users ka traffic bhi same pool se jaata hai). Exponential
//      backoff se short-lived blocks recover ho jaate hain.
// ---------------------------------------------------------------------------
function fetchWithRetry(url, options, maxRetries) {
  maxRetries = maxRetries || 3;
  var resp;
  for (var attempt = 0; attempt <= maxRetries; attempt++) {
    resp = UrlFetchApp.fetch(url, options);
    var code = resp.getResponseCode();
    if (code !== 429 && code < 500) {
      return resp;
    }
    if (attempt === maxRetries) {
      return resp; // retries khatam — asli response wapas (client ko dikhega)
    }
    // backoff: 1s, 2s, 4s (+ thoda random jitter taaki ek saath retry na karein)
    Utilities.sleep((Math.pow(2, attempt) * 1000) + Math.floor(Math.random() * 800));
  }
  return resp;
}


// ---------------------------------------------------------------------------
// 2) Router — target query param se provider + path nikalo
//    NOTE: GAS editor ke Run button se direct call karne par `e` undefined
//    hota hai — isliye har jagah safe default diye hain.
// ---------------------------------------------------------------------------
function handleRequest(e) {
  e = e || {};
  e.parameter = e.parameter || {};


  var target = e.parameter.target || "";
  var parts = target.split("/").filter(Boolean);


  if (!parts.length) {
    return jsonResponse({
      ok: true,
      message: "SmartRotator Google Apps Script Gateway ✓",
      providers: Object.keys(PROVIDERS),
      example: "?target=zen/v1/chat/completions",
    });
  }


  var providerName = parts[0].toLowerCase();
  var provider = PROVIDERS[providerName];
  if (!provider) {
    return jsonResponse({ error: "Unknown provider: " + providerName }, 404);
  }


  // upstream path — target me jo bhi path tha wo forward karo
  // e.g. target=zen/v1/chat/completions → /v1/chat/completions
  var upstreamPath = "/" + parts.slice(1).join("/");
  var upstreamUrl = provider.host + upstreamPath;


  // request body (POST) — doPost me e.postData.contents milta hai
  var payload = "";
  var method = "GET";
  if (e.postData && e.postData.contents) {
    payload = e.postData.contents;
    method = "POST";
  } else if (e.parameter.body) {
    // GET + ?body= (URL-encoded) fallback
    payload = e.parameter.body;
    method = "POST";
  }


  // headers — client ke auth/headers forward karo
  var headers = {
    "Content-Type": "application/json",
    "User-Agent": pick(USER_AGENTS),
    "Accept": "application/json, text/event-stream, text/plain, */*",
  };


  // OPENCODE ZEN FIX: official CLI headers inject karo (MissingSessionID se
  // bachne ke liye). Ye headers sirf zen upstream ke liye, baaki providers
  // pe browser fingerprint hi theek hai.
  if (providerName === "zen") {
    var ocHeaders = buildOpencodeHeaders();
    for (var k in ocHeaders) {
      headers[k] = ocHeaders[k];
    }
  }


  // Authorization / x-api-key header ko forward karo (agar client ne diya)
  // NOTE: GAS web app ko client ke HTTP headers NAHI milte — ye sirf query
  // param (auth/authorization) se aayega.
  var authHeader = e.parameter.auth || e.parameter.authorization;


  // GEMINI SPECIAL: SmartRotator ka gemini provider `params={"key": ...}`
  // use karta hai — GAS tak wo query param `key=` me aata hai. Gemini REST
  // API Authorization header nahi, `x-goog-api-key` header (ya ?key=) maangta
  // hai. Isliye key ko x-goog-api-key header me bhejo (rotation preserve
  // hoti hai — har request me SmartRotator apni key bhejega).
  var geminiKey = e.parameter.key || "";
  if (providerName === "gemini" && geminiKey) {
    headers["x-goog-api-key"] = geminiKey;
    authHeader = null; // Authorization bhejna mat — Gemini ko confuse karega
  }


  if (authHeader) {
    headers["Authorization"] = authHeader;
  }


  var options = {
    method: method,
    headers: headers,
    muteHttpExceptions: true,
    fetchTimeoutSeconds: 60,
    // follow redirects
    followRedirects: true,
  };


  if (method === "POST" && payload) {
    options.payload = payload;
  }


  try {
    var resp = fetchWithRetry(upstreamUrl, options, 3);
    var code = resp.getResponseCode();
    var text = resp.getContentText();


    // SmartRotator/OpenAI client ko PLAIN OpenAI-JSON chahiye (wrapped nahi).
    // 2xx pe seedha upstream body return karo — client seedha parse karega.
    if (code >= 200 && code < 300) {
      var okOut = ContentService.createTextOutput(text);
      okOut.setMimeType(ContentService.MimeType.JSON);
      return okOut;
    }


    // Non-2xx (400/401/429/500) — GAS status codes bhej nahi sakta (hamesha
    // 200 return karta hai), isliye wrapped response: client ko lagta hai ki
    // request ho gayi, par body me asli upstream error dikh raha hai.
    return ContentService.createTextOutput(
      JSON.stringify({
        __proxy_status: code,
        __proxy_provider: providerName,
        __proxy_error: text,
      })
    ).setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return jsonResponse({ error: "Gateway upstream error: " + err }, 502);
  }
}


function jsonResponse(obj, status) {
  var out = ContentService.createTextOutput(JSON.stringify(obj));
  out.setMimeType(ContentService.MimeType.JSON);
  return out;
}


// ---------------------------------------------------------------------------
// 4) TEST FUNCTION — GAS editor me Run dabao, seedha zen check hoga
//    (deploy ki zaroorat nahi, bas Zen key yahan daalo)
// ---------------------------------------------------------------------------
function testZen() {
  var ZEN_KEY = "PASTE_ZEN_KEY_HERE"; // apni zen key yahan daalo


  if (ZEN_KEY.indexOf("PASTE_") === 0) {
    Logger.log("❌ Pehle ZEN_KEY me apni key daalo (Code.gs me testZen function)");
    return;
  }


  var url = "https://opencode.ai/zen/v1/chat/completions";
  var headers = {
    "Authorization": ZEN_KEY,
    "Content-Type": "application/json",
  };

  // FIX: testZen bhi opencode headers bheje — warna MissingSessionID aayega
  var ocHeaders = buildOpencodeHeaders();
  for (var k2 in ocHeaders) {
    headers[k2] = ocHeaders[k2];
  }

  var options = {
    method: "POST",
    headers: headers,
    payload: JSON.stringify({
      model: "big-pickle",
      messages: [{ role: "user", content: "say hi" }],
      max_tokens: 50,
    }),
    muteHttpExceptions: true,
    fetchTimeoutSeconds: 60,
  };


  try {
    var resp = UrlFetchApp.fetch(url, options);
    var code = resp.getResponseCode();
    var text = resp.getContentText();
    Logger.log("✅ Zen response code: " + code);
    Logger.log("Response: " + text.slice(0, 300));
  } catch (err) {
    Logger.log("❌ Error: " + err);
  }
}


// ---------------------------------------------------------------------------
// 5) Web App entry points
// ---------------------------------------------------------------------------
function doGet(e) {
  return handleRequest(e);
}


function doPost(e) {
  return handleRequest(e);
}