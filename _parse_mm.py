
# -*- coding: utf-8 -*-
"""
解析 .mm 文件，提取所有非末端节点，并重写 sample_feedback_input.csv
使得 node_id 字段引用真实的非末端节点 ID。
"""
import xml.etree.ElementTree as ET
import csv, random, re, os

BASE = os.path.dirname(__file__)
MM_PATH = os.path.join(BASE, "星铁中国大陆网络社区生态地图.mm")
CSV_IN  = os.path.join(BASE, "data", "inbox", "sample_feedback_input.csv")
CSV_OUT = os.path.join(BASE, "data", "inbox", "sample_feedback_input.csv")  # 覆盖写回

tree = ET.parse(MM_PATH)
root = tree.getroot()

non_leaf = []   # list of (id, text, depth)

def traverse(node, depth=0):
    kids = list(node)
    nid  = node.get('ID', '')
    text = node.get('TEXT', '')
    # 清理 HTML 实体标签
    text = re.sub(r'&amp;lt;br&amp;gt;|&lt;br&gt;', '', text).strip()
    if kids:
        non_leaf.append((nid, text, depth))
        for c in kids:
            traverse(c, depth + 1)

traverse(root[0], 0)   # root[0] 是地图主节点

print(f"[parse_mm] 共找到非末端节点 {len(non_leaf)} 个")
for d, (nid, text, depth) in enumerate(non_leaf[:20]):
    print(f"  depth={depth}  id={nid}  text={text[:50]}")

# ── 读取 CSV ──────────────────────────────────────────────
with open(CSV_IN, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

print(f"[parse_mm] CSV 共 {len(rows)} 行")

# ── 逐行替换 node_id ──────────────────────────────────────
# 排除根节点（depth==0），只取 depth>=1 的真实分类节点
candidate_nodes = [(nid, text, depth) for (nid, text, depth) in non_leaf if depth >= 1]
print(f"[parse_mm] 候选分类节点 {len(candidate_nodes)} 个")

random.seed(42)
new_rows = []
for row in rows:
    picked = random.choice(candidate_nodes)
    row['node_id'] = picked[0]
    new_rows.append(row)

# ── 写回 CSV ──────────────────────────────────────────────
with open(CSV_OUT, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)

print(f"[parse_mm] 已写回 {CSV_OUT}")
print("[parse_mm] 前5行 node_id 示例：")
for row in new_rows[:5]:
    print(f"  {row['record_id']}  node_id={row['node_id']}")
