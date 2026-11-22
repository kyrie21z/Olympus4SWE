export interface Point {
  x: number;
  y: number;
}

const round = (value: number): number => Math.round(value * 100) / 100;

/** Drop repeated points and the middle point of collinear runs. */
export function simplify(points: Point[]): Point[] {
  const unique = points.filter(
    (point, index) =>
      index === 0 ||
      Math.abs(point.x - points[index - 1].x) > 0.01 ||
      Math.abs(point.y - points[index - 1].y) > 0.01,
  );
  return unique.filter((point, index) => {
    if (index === 0 || index === unique.length - 1) return true;
    const prev = unique[index - 1];
    const next = unique[index + 1];
    const cross = (point.x - prev.x) * (next.y - point.y) - (point.y - prev.y) * (next.x - point.x);
    return Math.abs(cross) > 0.01;
  });
}

function toward(from: Point, to: Point, distance: number): Point {
  const length = Math.hypot(to.x - from.x, to.y - from.y) || 1;
  return {
    x: from.x + ((to.x - from.x) / length) * distance,
    y: from.y + ((to.y - from.y) / length) * distance,
  };
}

/** Orthogonal polyline with softly rounded corners (quadratic joins, radius clamped per leg). */
export function roundedPolylinePath(points: Point[], radius = 6): string {
  const pts = simplify(points);
  if (pts.length < 2) return "";
  let d = `M${round(pts[0].x)},${round(pts[0].y)}`;
  for (let i = 1; i < pts.length - 1; i += 1) {
    const prev = pts[i - 1];
    const corner = pts[i];
    const next = pts[i + 1];
    const r = Math.min(
      radius,
      Math.hypot(corner.x - prev.x, corner.y - prev.y) / 2,
      Math.hypot(next.x - corner.x, next.y - corner.y) / 2,
    );
    const a = toward(corner, prev, r);
    const b = toward(corner, next, r);
    d += ` L${round(a.x)},${round(a.y)} Q${round(corner.x)},${round(corner.y)} ${round(b.x)},${round(b.y)}`;
  }
  const last = pts[pts.length - 1];
  d += ` L${round(last.x)},${round(last.y)}`;
  return d;
}
