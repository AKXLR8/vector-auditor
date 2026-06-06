import asyncio
import os
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
import asyncpg

async def go():
    url = os.environ['DATABASE_URL']
    if url.startswith('postgresql+asyncpg://'):
        url = url.replace('postgresql+asyncpg://', 'postgresql://')
    from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
    p = urlparse(url); qs = parse_qs(p.query, keep_blank_values=True)
    qs.pop('sslmode', None); qs.pop('ssl', None)
    url = urlunparse(p._replace(query=urlencode(qs, doseq=True)))
    conn = await asyncpg.connect(url, ssl=True)
    tables = [r['table_name'] for r in await conn.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")]
    for t in tables:
        print(f"\n=== {t} ===")
        for r in await conn.fetch("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = $1 ORDER BY ordinal_position
        """, t):
            print(f"  {r['column_name']:30} {r['data_type']:30} nullable={r['is_nullable']} default={r['column_default']}")
    await conn.close()

asyncio.run(go())
