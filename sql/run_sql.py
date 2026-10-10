"""Build the SQLite database and run the Part 1 reports.

The sqlite3 command-line shell is not required: this uses Python's built-in
sqlite3 module, so it works anywhere Python does.

Usage (from the repo root):
    python sql/run_sql.py                  # rebuild DB, run reports, print results
    python sql/run_sql.py --refresh-reports  # also rewrite the "-- |" output
                                             # comment blocks inside reports.sql
"""
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parent
DB_PATH = SQL_DIR / "mamaearth.db"
HEADER = re.compile(r"^-- ===== \(([a-i])\) (.*) =====\s*$")


def fmt_table(cols, rows):
    """Return the result as '| a | b |' lines (header first)."""
    out = ["| " + " | ".join(cols) + " |"]
    out += ["| " + " | ".join("NULL" if v is None else str(v) for v in r) + " |" for r in rows]
    return out


def run_statement(conn, sql):
    cur = conn.execute(sql)
    if cur.description:
        return [d[0] for d in cur.description], cur.fetchall()
    return None, None


def split_statements(body_lines):
    """Split lines into [(leading_comment_lines, sql_text)] on ';'."""
    stmts, comments, buf = [], [], []
    for line in body_lines:
        if not buf and (line.strip() == "" or line.lstrip().startswith("--")):
            if line.strip():
                comments.append(line)
            continue
        buf.append(line)
        if line.rstrip().endswith(";"):
            stmts.append((comments, "\n".join(buf)))
            comments, buf = [], []
    return stmts


def build_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    subprocess.run([sys.executable, str(SQL_DIR / "generate_seed.py")], check=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
    seed = (SQL_DIR / "seed_data.sql").read_text(encoding="utf-8")
    # executescript can't return rows, so run the verification SELECTs separately
    stmts = [s for _, s in split_statements(seed.splitlines())]
    for s in stmts:
        cols, rows = run_statement(conn, s)
        if cols:
            print("\n".join(fmt_table(cols, rows)), "\n")
    conn.commit()
    return conn


def main(refresh):
    conn = build_db()
    src = (SQL_DIR / "reports.sql").read_text(encoding="utf-8").splitlines()

    # Split into preamble + sections, dropping any old output blocks
    sections, current, preamble = [], None, []
    for line in src:
        if line.startswith("-- |") or line.strip() == "-- Output:":
            continue  # old output block: it is regenerated below
        m = HEADER.match(line)
        if m:
            current = {"id": m.group(1), "title": m.group(2), "header": line, "lines": []}
            sections.append(current)
        elif current is None:
            preamble.append(line)
        else:
            current["lines"].append(line)

    rebuilt = list(preamble)
    for sec in sections:
        print(f"===== ({sec['id']}) {sec['title']} =====")
        rebuilt.append(sec["header"])
        for comments, sql in split_statements(sec["lines"]):
            cols, rows = run_statement(conn, sql)
            rebuilt.extend(comments)
            if cols:
                block = fmt_table(cols, rows)
                print("\n".join(block), "\n")
                rebuilt.append("-- Output:")
                rebuilt.extend("-- " + b for b in block)
            rebuilt.append(sql)
            rebuilt.append("")
    conn.commit()
    conn.close()

    if refresh:
        text = "\n".join(rebuilt).rstrip() + "\n"
        (SQL_DIR / "reports.sql").write_text(text, encoding="utf-8")
        print("reports.sql output comments refreshed.")


if __name__ == "__main__":
    main("--refresh-reports" in sys.argv)
