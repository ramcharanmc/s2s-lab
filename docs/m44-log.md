# M44 run log: the check before the repository is made public

What happened, in order. The earlier logs: `docs/m43-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-20 | read-only scans of the tracked files and the whole history, laptop | no secret, no audio or weights, no large file; no local path or e-mail in any file now; the author's name and e-mail on all 201 commits; 22 added lines in old commits with the author's local path (section 1) | — | — |
| 2 | 10-20 | each model's and dataset's licence, read from its own card | all named in one table in the README (section 2) | — | — |
| 3 | 10-20 | — | the author's choice: publish with the history as it is (section 3) | README's licences, records | this commit |

## 1. The scans (row 1)

| What | Where it was looked for | Found |
|---|---|---|
| Secrets: access tokens (Hugging Face, OpenAI, AWS, GitHub, Slack, Modal), private keys, a value given to `DEMO_ACCESS_CODE` | every tracked file, and every line ever added in any commit | **none**. The one match is the instruction in `modal_app/demo.py` that reads the code from a hidden prompt; it holds no value |
| Files that look like secrets (`.env`, keys, credentials) | every path ever committed | **none** |
| Audio, model weights, data files (`.wav`, `.mp3`, `.webm`, `.pt`, `.safetensors`, `.onnx`, `.parquet`, ...) | every path ever committed | **none**, now or ever |
| Large files | every blob ever committed | the largest 1.2 MB (`runs/r4-gpu/pr017.json`); the 437 tracked files 13.4 MB together |
| A local path (`C:\Users\...`, `/mnt/c/...`, `/home/...`) | the tracked files now | **none** |
| | every line ever added | **22 lines**, removed from the files in M27 (`docs/m27-log.md`, entries 29 and 33), still in the history |
| An e-mail address in a file | the tracked files now, and every line ever added | **none** |
| The author's e-mail | the commits' author and committer fields | **on all 201 commits**, with the author's name |

## 2. The licences (row 2)

Read on 2026-10-20 from each source's own card (Hugging Face's licence field, a repository's licence file, a
package's own text). Two to know: **openSMILE** (the affect tagger's feature extractor) is free for private,
research and educational use only, not for a commercial product; and **Whisper**'s Hugging Face card lists
Apache-2.0 where OpenAI released it under MIT. EdAcc and DailyTalk are share-alike (CC BY-SA 4.0); PersonaPlex is
under NVIDIA's Open Model License. Nothing here is redistributed but text: the eval set's and the probe sets'
transcripts and sentences, which keep their datasets' licences, as the README now says. The table is in the
README, "Licences".

## 3. The author's choice (row 3)

Of three options (publish with the history as it is; rewrite the history to remove the e-mail and the old paths;
keep the repository private), **publish with the history as it is**. A rewrite would change all 201 commit hashes,
and the registrations are dated by their commits: the run files, the logs and the build log cite those hashes
hundreds of times. The author's name as the author is intended.

**What is left, the author's, in GitHub's settings:**
1. **Settings, Emails:** "Keep my email addresses private". GitHub shows the private address,
   `<id>+ramcharanmc@users.noreply.github.com`.
2. In WSL: `git config --global user.email "<that address>"`, so later commits carry it. The earlier 201 keep the
   Gmail address.
3. Then, if wanted, "Block command line pushes that expose my email", in the same place.
4. **Settings, General, Danger Zone: Change visibility, Public.** The CI badges then show to everyone, and Actions
   minutes are free for a public repository.
