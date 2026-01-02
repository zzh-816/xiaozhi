#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sqlite3
import os

db_path = 'data/multi_tier_memory/memory_CA_F9_55_4C_4F_3D.db'

if not os.path.exists(db_path):
    print(f"数据库文件不存在: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询工作记忆（注意：数据库中的user_id是原始格式，带冒号）
cursor.execute("""
    SELECT role, content, timestamp 
    FROM working_memory 
    WHERE user_id = ? 
    ORDER BY timestamp DESC 
    LIMIT 20
""", ('CA:F9:55:4C:4F:3D',))

rows = cursor.fetchall()
print(f"工作记忆记录 (最近20条，共 {len(rows)} 条):")
print("=" * 80)
for i, row in enumerate(rows, 1):
    role, content, timestamp = row
    content_preview = content[:60] + "..." if len(content) > 60 else content
    print(f"{i}. [{timestamp}] {role}: {content_preview}")

# 查询长期记忆（注意：数据库中的user_id是原始格式，带冒号）
cursor.execute("""
    SELECT memory_type, content, updated_at 
    FROM long_term_memory 
    WHERE user_id = ?
""", ('CA:F9:55:4C:4F:3D',))

lt_rows = cursor.fetchall()
print(f"\n长期记忆记录 (共 {len(lt_rows)} 条):")
print("=" * 80)
for i, row in enumerate(lt_rows, 1):
    memory_type, content, updated_at = row
    content_preview = content[:60] + "..." if len(content) > 60 else content
    print(f"{i}. [{updated_at}] {memory_type}: {content_preview}")

conn.close()

