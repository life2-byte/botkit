"""python -m botkit.memory  ->  prints the Supabase SQL (tables + grants)."""
import sys

from .supabase import schema_sql

print(schema_sql(sys.argv[1] if len(sys.argv) > 1 else "botkit_"))
