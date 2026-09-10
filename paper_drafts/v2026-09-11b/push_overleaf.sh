#!/bin/bash
# Build, verify, commit and push the paper to Overleaf. Usage: ./push_overleaf.sh "commit message"
set -e
cd "$(dirname "$0")"
MSG=${1:-"Update"}
bash ./build.sh > /tmp/build.log 2>&1 || { echo "BUILD FAILED"; grep -E '^!' /tmp/build.log | head -5; exit 1; }
U=$(grep -cE 'undefined' iclr2027_conference.log || true)
[ "$U" != "0" ] && { echo "UNDEFINED REFERENCES: $U"; grep -E 'undefined' iclr2027_conference.log | head -5; exit 1; }
PAGES=$(python3 -c "import re;s=open('iclr2027_conference.log',encoding='latin-1').read();m=re.findall(r'Output written on .*? \((\d+) pages',s);print(m[-1] if m else '?')")
git add -A sections iclr2027_conference.tex iclr2027_conference.bib push_overleaf.sh build.sh 2>/dev/null
if git diff --cached --quiet; then echo "nothing to commit (pages $PAGES)"; exit 0; fi
git -c user.name='Zhizhe Zhang' -c user.email='zz5070@nyu.edu' commit -q -m "$MSG"
git -c credential.helper='store --file /Users/zzz/.git-credentials-overleaf' fetch -q origin 2>/dev/null
git -c user.name='Zhizhe Zhang' -c user.email='zz5070@nyu.edu' rebase -q origin/main
git -c credential.helper='store --file /Users/zzz/.git-credentials-overleaf' push -q origin main 2>&1 | grep -vE 'failed to (get|store)' || true
echo "pushed: $(git log --oneline -1) (pages $PAGES)"
