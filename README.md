# public-posted-images

Public image host for Echo-OS published posts. Each subfolder is a channel.

| Folder | Channel | Posts to |
|---|---|---|
| `critter/` | @DEARDANES | Facebook Page "Dear Danes" via Buffer |

Images in this repo are consumed at publish time by downstream social schedulers
(currently Buffer → Facebook). The raw.githubusercontent.com URL for each image
is written back to that channel's articles ledger for Buzz to look up.

**Do not delete images in this repo without confirming no live post references them.**
