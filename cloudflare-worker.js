/**
 * SmartRotator — Cloudflare Worker API Gateway (multi-provider)
 * =============================================================
 *
 * Kya karta hai:
 *   1. Ek hi Worker se saare providers (Gemini, Groq, OpenRouter, NVIDIA,
 *      Zen) ka base_url point kar sakte ho — path se provider detect hota hai.
 *   2. Har request pe ALAG User-Agent + browser-like headers lagta hai
 *      (kisi aur OS/browser se aa rahi lagti hai — Windows/MacOS/Linux,
 *      Chrome/Edge/Firefox/Safari random rotate hota hai).
 *   3. Real client IP / X-Forwarded-For headers STRIP hoti hain — provider
 *      ko sirf Cloudflare edge IP dikhta hai (har request alag CF edge
 *      location se ho sakti hai).
 *
 * Deploy:
 *   Cloudflare Dashboard → Workers & Pages → Create Worker → ye code paste
 *   → Deploy. Phir SmartRotator ke 🔌 Providers tab me base_url daalo:
 *
 *     gemini     → https://<worker>.workers.dev/gemini/v1beta
 *     groq       → https://<worker>.workers.dev/groq/openai/v1
 *     openrouter → https://<worker>.workers.dev/openrouter/api/v1
 *     nvidia     → https://<worker>.workers.dev/nvidia/v1
 *     zen        → https://<worker>.workers.dev/zen/v1
 *
 * (aur har provider ki keys wahi rahengi — worker bas route/spoof karta hai)
 *
 * Note: ye rate-limit bypass nahi karta, bas IP diversity + fingerprint
 * rotation deta hai. Free tier: 100k requests/day, 10ms CPU per request.
 */

// ---------------------------------------------------------------------------
// 1) Provider routes — provider ka REAL upstream host
// ---------------------------------------------------------------------------
const PROVIDERS = {
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
  zen: {
    host: "https://opencode.ai/zen", // NOTE: zen ka real base /zen/v1 hai,
    label: "OpenCode Zen",           // isliye host ke saath /zen rakha (route
  },                                 // /zen/v1 se host+path = .../zen/v1)
};

// ---------------------------------------------------------------------------
// 2) Fingerprint rotation — real browser/OS combos
//    Har request pe random pick hota hai → provider ko alag OS+browser dikhta
// ---------------------------------------------------------------------------
const FINGERPRINTS = [
  // Windows 11 + Chrome
  {
    ua: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    platform: '"Windows"',
    brands: '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
    mobile: "?0",
  },
  // Windows 11 + Edge
  {
    ua: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.2592.87",
    platform: '"Windows"',
    brands: '"Chromium";v="126", "Microsoft Edge";v="126", "Not.A/Brand";v="99"',
    mobile: "?0",
  },
  // macOS + Safari
  {
    ua: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    platform: '"macOS"',
    brands: '"Not/A)Brand";v="8", "Safari";v="17.5", "AppleWebKit";v="605"',
    mobile: "?0",
  },
  // macOS + Chrome
  {
    ua: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    platform: '"macOS"',
    brands: '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
    mobile: "?0",
  },
  // Linux + Chrome
  {
    ua: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    platform: '"Linux"',
    brands: '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
    mobile: "?0",
  },
  // Linux + Firefox
  {
    ua: "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    platform: '"Linux"',
    brands: '"Not/A)Brand";v="8", "Firefox";v="127.0", "Gecko";v="20100101"',
    mobile: "?0",
  },
  // Android + Chrome (mobile)
  {
    ua: "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro Build/AP1A.240505.005) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.71 Mobile Safari/537.36",
    platform: '"Android"',
    brands: '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
    mobile: "?1",
  },
  // iPhone + Safari (iOS)
  {
    ua: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    platform: '"iOS"',
    brands: '"Not/A)Brand";v="8", "Safari";v="17.5", "AppleWebKit";v="605"',
    mobile: "?1",
  },
];

// random locale (Accept-Language) — har request pe alag region dikhe
const LOCALES = [
  "en-US,en;q=0.9",
  "en-GB,en;q=0.9",
  "en-IN,en;q=0.9",
  "de-DE,de;q=0.9",
  "fr-FR,fr;q=0.9",
  "ja-JP,ja;q=0.9",
  "es-ES,es;q=0.9",
  "pt-BR,pt;q=0.9",
];

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function buildBrowserHeaders(base) {
  const fp = pick(FINGERPRINTS);
  const h = new Headers(base);
  h.set("User-Agent", fp.ua);
  h.set("Accept", "application/json, text/event-stream, text/plain, */*");
  h.set("Accept-Language", pick(LOCALES));
  h.set("Accept-Encoding", "gzip, deflate, br");
  h.set("sec-ch-ua", fp.brands);
  h.set("sec-ch-ua-mobile", fp.mobile);
  h.set("sec-ch-ua-platform", fp.platform);
  h.set("Sec-Fetch-Dest", "empty");
  h.set("Sec-Fetch-Mode", "cors");
  h.set("Sec-Fetch-Site", "cross-site");
  // kisi bhi request ko "browser client" jaisa banane ke liye cache/dnt
  h.set("Cache-Control", "no-cache");
  h.set("DNT", "1");
  return h;
}

// ---------------------------------------------------------------------------
// 3) Privacy — real client identity ko provider tak NAHI pahunchne do
// ---------------------------------------------------------------------------
const PRIVACY_HEADERS = [
  "cf-connecting-ip",
  "cf-ray",
  "cf-visitor",
  "cf-ipcountry",
  "x-forwarded-for",
  "x-forwarded-proto",
  "x-forwarded-host",
  "x-real-ip",
  "true-client-ip",
  "forwarded",
  "via",
  "x-request-id",
];

function stripPrivacyHeaders(headers) {
  for (const name of PRIVACY_HEADERS) {
    headers.delete(name);
  }
  // apni taraf se ek generic X-Forwarded-For set karo (kuch providers bina
  // iske reject karte hain) — par koi real IP nahi, bas Cloudflare edge
  headers.set("X-Forwarded-For", "203.0.113.10"); // TEST-NET-3 dummy
  return headers;
}

// ---------------------------------------------------------------------------
// 4) Main handler
// ---------------------------------------------------------------------------
export default {
  async fetch(request) {
    const url = new URL(request.url);
    const parts = url.pathname.split("/").filter(Boolean);

    if (!parts.length) {
      return new Response(
        "SmartRotator Cloudflare Gateway ✓\nRoute format: /PROVIDER/rest/of/path\n\n" +
          "Providers: " + Object.keys(PROVIDERS).join(", ") + "\n" +
          "Example:   /gemini/v1beta/models (live models fetch)\n",
        { status: 200, headers: { "Content-Type": "text/plain; charset=utf-8" } }
      );
    }

    const providerName = parts[0].toLowerCase();
    const provider = PROVIDERS[providerName];
    if (!provider) {
      return new Response("Unknown provider: " + providerName, { status: 404 });
    }

    // real upstream URL — provider ke host pe path+query forward
    const upstreamPath = "/" + parts.slice(1).join("/");
    const upstreamUrl = provider.host + upstreamPath + url.search;

    // headers: real request ke useful wale lo (Authorization, Content-Type)
    // phir browser fingerprint lagao + privacy strip karo
    let headers = new Headers(request.headers);
    headers = buildBrowserHeaders(headers);
    headers = stripPrivacyHeaders(headers);

    const init = { method: request.method, headers, redirect: "follow" };
    if (!["GET", "HEAD"].includes(request.method)) {
      init.body = request.body;
    }

    try {
      const upstream = await fetch(upstreamUrl, init);

      // response ko SSE/JSON bina change bhejo
      const outHeaders = new Headers(upstream.headers);
      outHeaders.set("Access-Control-Allow-Origin", "*");
      // streaming/SSE ke liye buffering disable (aisa hi CF best practice hai)
      outHeaders.set("Cache-Control", "no-cache, no-store");

      return new Response(upstream.body, {
        status: upstream.status,
        statusText: upstream.statusText,
        headers: outHeaders,
      });
    } catch (err) {
      return new Response(
        "Gateway upstream error: " + (err && err.message ? err.message : err),
        { status: 502, headers: { "Content-Type": "text/plain; charset=utf-8" } }
      );
    }
  },
};
