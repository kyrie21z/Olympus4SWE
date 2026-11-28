const privateLabel = /(?:sk-|gh[pousr]_|github_pat_|akia[0-9a-z]{16}|api[_-]?key|bearer|password|secret|token[=_-])/i;

/** Short identifiers suitable for evidence views; exclude free text and credential-shaped strings. */
export function publicId(id: unknown): id is string {
  return typeof id === "string" && /^[\p{L}\p{N}_.-]{1,96}$/u.test(id) && !privateLabel.test(id);
}
