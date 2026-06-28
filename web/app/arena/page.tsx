import { redirect } from "next/navigation";

// ELO Arena hidden in v0.1.5: it shipped as a hardcoded all-1200-ELO
// placeholder in v0.1.4. Real head-to-head matches land in v0.2; until then
// the Benchmark page is the source of truth. We keep the route around so old
// links don't 404.
export default function ArenaRedirect() {
  redirect("/benchmark");
}
