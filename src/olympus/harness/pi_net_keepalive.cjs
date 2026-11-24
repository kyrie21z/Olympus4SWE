"use strict";

// Preloaded into Pi's Node.js subprocess via NODE_OPTIONS=--require so long
// reasoning/streaming requests keep TCP connections alive across 60s NAT/gateway
// idle windows and never hit short body/headers read timeouts.

(() => {
  const MIN_READ_TIMEOUT_MS = 300_000;
  const DEFAULT_READ_TIMEOUT_MS = 600_000;
  const DEFAULT_KEEPALIVE_MS = 20_000;

  function parsePositiveInt(raw, fallback) {
    const parsed = Number.parseInt(String(raw ?? ""), 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  }

  const keepAliveMs = parsePositiveInt(
    process.env.OLYMPUS_PI_TCP_KEEPALIVE_MS,
    DEFAULT_KEEPALIVE_MS
  );
  const readTimeoutMs = Math.max(
    MIN_READ_TIMEOUT_MS,
    parsePositiveInt(process.env.OLYMPUS_PI_READ_TIMEOUT_MS, DEFAULT_READ_TIMEOUT_MS)
  );

  // 1. Enable TCP keep-alive on every outbound net.Socket connection.
  try {
    const net = require("net");
    if (net && net.Socket && typeof net.Socket.prototype.connect === "function") {
      const origConnect = net.Socket.prototype.connect;
      if (!origConnect.__olympusKeepAlivePatched) {
        const patched = function olympusKeepAliveConnect(...args) {
          const result = origConnect.apply(this, args);
          try {
            if (typeof this.setKeepAlive === "function") {
              this.setKeepAlive(true, keepAliveMs);
            }
          } catch (_) {
            // Best-effort only; never disrupt socket connection setup.
          }
          return result;
        };
        patched.__olympusKeepAlivePatched = true;
        net.Socket.prototype.connect = patched;
      }
    }
  } catch (_) {
    // Ignore on non-Node or restricted runtimes.
  }

  // 2. Ensure Node's undici global fetch dispatcher enforces >= 300s read
  // timeouts and TCP keep-alive while preserving EnvHttpProxyAgent proxy rules.
  try {
    const sym = Symbol.for("undici.globalDispatcher.1");
    const current = globalThis[sym];
    if (current && typeof current.constructor === "function") {
      const DispatcherCtor = current.constructor;
      const next = new DispatcherCtor({
        headersTimeout: readTimeoutMs,
        bodyTimeout: readTimeoutMs,
        connect: {
          keepAlive: true,
          keepAliveInitialDelay: keepAliveMs,
        },
      });
      globalThis[sym] = next;
    }
  } catch (_) {
    // Keep default dispatcher if runtime differs.
  }
})();
