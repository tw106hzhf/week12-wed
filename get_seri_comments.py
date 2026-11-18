import csv
import json
import re

# Regex to match variations of Seri's name (Seri, Se-ri, Se ri, Yoon Seri, Yoon Se-ri, etc.)
# \b ensures we don't accidentally match words like "series" or "serious"
SERI_PATTERN = re.compile(r"\b(?:yoon[\s-]*)?se[\s-]*ri\b", re.IGNORECASE)


def extract_matching_comments(comments, title, results):
    """Recursively traverses comments and nested replies."""
    for comment in comments:
        body = comment.get("comment_body", "")
        word_count = len(body.split())

        if 20 <= word_count <= 100 and SERI_PATTERN.search(body):
            results.append({
                "timestamp": comment.get("created_utc"),
                "submission title": title,
                "comment_text": body,
            })

        # Recursively check nested replies if they exist
        replies = comment.get("replies")
        if replies:
            extract_matching_comments(replies, title, results)


def process_threads(input_file, output_csv):
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        # Handles both a single JSON list of threads OR JSONL (one JSON object per line)
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            threads = json.load(f)
        else:
            threads = [json.loads(line) for line in f if line.strip()]

    for thread in threads:
        title = thread.get("title", "")
        comments = thread.get("comments", [])
        extract_matching_comments(comments, title, results)

    # Write results to CSV
    fieldnames = ["timestamp", "submission title", "comment_text"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Saved {len(results)} matching comments to {output_csv}")


if __name__ == "__main__":
    # Replace 'threads.json' with your actual filename
    process_threads("cloy_chronological.jsonl", "seri_comments.csv")