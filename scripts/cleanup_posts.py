#!/usr/bin/env python3
"""
Post archive cleanup.

Generated posts are stored flat as posts/ai-ml-weekly-YYYY-MM-DD.md. This script
deletes posts older than a retention window so the archive doesn't grow without
bound. It is meant to run monthly. The persistent dedup record (posts/.seen.json)
is never touched, so already-featured stories stay suppressed even after their
posts are removed.

Usage:
    python scripts/cleanup_posts.py                 # delete posts older than KEEP_DAYS (default 60)
    python scripts/cleanup_posts.py --keep-days 90
    python scripts/cleanup_posts.py --dry-run       # show what would be deleted

Environment:
    KEEP_DAYS   Delete posts older than this many days (default 60, ~2 months / 8 posts).
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta

# Ensure emoji/status output doesn't crash on non-UTF-8 consoles (e.g. Windows cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

POSTS_DIR = "posts"
POST_RE = re.compile(r"^ai-ml-weekly-(\d{4}-\d{2}-\d{2})\.md$")
DEFAULT_KEEP_DAYS = 60  # ~2 months, keeps roughly the last 8 weekly posts


def find_posts(posts_dir=POSTS_DIR):
    """Return list of (filepath, post_date, filename) for every post file."""
    if not os.path.isdir(posts_dir):
        return []
    posts = []
    for filename in os.listdir(posts_dir):
        match = POST_RE.match(filename)
        if not match:
            continue
        try:
            post_date = datetime.strptime(match.group(1), "%Y-%m-%d")
        except ValueError:
            continue
        posts.append((os.path.join(posts_dir, filename), post_date, filename))
    return posts


def cleanup(keep_days=DEFAULT_KEEP_DAYS, posts_dir=POSTS_DIR, dry_run=False):
    """Delete posts older than ``keep_days`` days."""
    if keep_days < 0:
        print("keep-days must be >= 0; aborting.")
        return 1

    cutoff = datetime.now() - timedelta(days=keep_days)
    posts = find_posts(posts_dir)
    to_delete = sorted((p for p in posts if p[1] < cutoff), key=lambda x: x[1])

    if not to_delete:
        print(f"📊 No cleanup needed: {len(posts)} post(s), none older than "
              f"{keep_days} days.")
        return 0

    verb = "Would delete" if dry_run else "Deleted"
    for filepath, _post_date, filename in to_delete:
        if not dry_run:
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"❌ Error deleting {filename}: {e}")
                continue
        print(f"🗑️  {verb} old post: {filename}")

    kept = len(posts) - len(to_delete)
    print(f"📊 Cleanup complete: kept {kept} recent post(s), "
          f"{'would remove' if dry_run else 'removed'} {len(to_delete)} "
          f"older than {keep_days} days.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Delete old posts past the retention window.")
    parser.add_argument(
        "--keep-days", type=int,
        default=int(os.getenv("KEEP_DAYS", DEFAULT_KEEP_DAYS)),
        help="Delete posts older than this many days (default 60).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without deleting anything.",
    )
    args = parser.parse_args()
    return cleanup(keep_days=args.keep_days, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
