Output "docs/demo.gif"

Set FontSize 18
Set Width 1280
Set Height 720
Set Padding 24
Set Theme "Atom"
Set Shell zsh

# vhs starts a fresh shell where the venv isn't on PATH. Activate first so
# `memory-arena` resolves. The activate is hidden so the very first visible
# frame is the typed command — the v0.1.4 GIF was flagged for showing a
# blank prompt as frame 0.
Hide
Type "source .venv/bin/activate"
Enter
Sleep 400ms
Type "clear"
Enter
Sleep 200ms
Show

Type "memory-arena report --corpus longmemeval-s 2>&1 | head -10"
Sleep 800ms
Enter
Sleep 5000ms

Type "ls memory_arena/data/results_snapshot/longmemeval-s_*_summary.json | sed 's|.*/||' | head -5"
Sleep 600ms
Enter
Sleep 3000ms

Type "python3 scripts/aggregate_bootstrap.py 2>&1 | head -10"
Sleep 600ms
Enter
Sleep 5000ms

Type "echo 'Hero chart at docs/hero.png'"
Sleep 400ms
Enter
Sleep 1500ms
