/**
 * Keep the reader's place across live updates: focus, open disclosures, inner scroll offsets and
 * the heading closest to the reading line under the sticky stage header.
 *
 * React keeps most nodes in place, but live projections can still grow or shrink content above
 * the reader. `capture()` runs before a commit and returns `restore()` for right after it.
 */

type Finder = () => Element | null | undefined;

const KEY_ATTRIBUTES = [
  "data-output-key",
  "data-task-id",
  "data-workflow-node",
  "data-tab",
  "data-run-id",
  "data-amb-input",
  "data-jump-task",
  "data-history-output",
  "data-review-disclosure",
  "aria-label",
];

/** A stable way back to `element` after a re-render: nearest id + keyed selector + index. */
function renderedElement(element: Element): Finder {
  const root = element.closest("[id]");
  if (!root) return () => null;
  if (root === element) return () => document.getElementById(root.id);
  const key = KEY_ATTRIBUTES.find((attribute) => element.hasAttribute(attribute));
  const tag = element.localName;
  const selector = key ? `${tag}[${key}="${CSS.escape(element.getAttribute(key) || "")}"]` : tag;
  const index = Array.from(root.querySelectorAll(selector)).indexOf(element);
  return () => document.getElementById(root.id)?.querySelectorAll(selector)[index];
}

export function captureReadingPosition(): () => void {
  const active = (document.activeElement as HTMLElement | null) || document.body;
  const findActive = renderedElement(active);
  const activeRow = active.closest("[data-output-key]");
  const findActiveRow = activeRow ? renderedElement(activeRow) : null;
  const scrollers = Array.from(document.querySelectorAll<HTMLElement>(".scroller"))
    .filter((element) => element.scrollTop || element.scrollLeft)
    .map((element) => ({
      find: renderedElement(element),
      top: element.scrollTop,
      left: element.scrollLeft,
    }));
  const openDetails = Array.from(document.querySelectorAll("details[open]"), renderedElement);
  const stage = document.getElementById("stage");
  const header = stage?.querySelector(".stage-head");
  if (!stage || !header) return () => undefined;
  const documentScroll = getComputedStyle(header).position !== "sticky";
  const scrollRoot = documentScroll ? document.scrollingElement || document.documentElement : stage;
  const pageTop = scrollRoot.scrollTop;
  const readingLine = Math.max(0, header.getBoundingClientRect().bottom);
  const anchors = pageTop
    ? Array.from(stage.querySelectorAll("[data-output-key], section[id], h2[id]")).filter((element) => {
        const box = element.getBoundingClientRect();
        return box.height && box.bottom > readingLine && box.top < window.innerHeight;
      })
    : [];
  const anchor = anchors.reduce<Element | undefined>(
    (closest, element) =>
      !closest ||
      Math.abs(element.getBoundingClientRect().top - readingLine) <
        Math.abs(closest.getBoundingClientRect().top - readingLine)
        ? element
        : closest,
    undefined,
  );
  const anchorTop = anchor?.getBoundingClientRect().top ?? 0;
  const findAnchor = anchor ? renderedElement(anchor) : null;

  return () => {
    openDetails.forEach((find) => {
      const details = find();
      if (details instanceof HTMLDetailsElement) details.open = true;
    });
    if (!active.isConnected && document.activeElement === document.body) {
      const target = (findActive() || findActiveRow?.()) as HTMLElement | null | undefined;
      target?.focus({ preventScroll: true });
    }
    for (const { find, top, left } of scrollers) {
      const element = find();
      if (element) element.scrollTo({ top, left, behavior: "instant" });
    }
    const target = findAnchor?.();
    if (documentScroll) scrollRoot.scrollTop = pageTop;
    if (target?.getClientRects().length) {
      scrollRoot.scrollTop += target.getBoundingClientRect().top - anchorTop;
    }
  };
}

/** Brief accent wash on a progress line; skipped for reduced motion or hidden elements. */
export function highlightProgress(element: HTMLElement | null): void {
  if (!element) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !element.getClientRects().length) {
    return;
  }
  element.getAnimations().forEach((animation) => animation.cancel());
  element.animate(
    [{ backgroundColor: "var(--accent-soft)" }, { backgroundColor: "var(--surface-raised)" }],
    { duration: 240, easing: "cubic-bezier(0.23, 1, 0.32, 1)" },
  );
}
