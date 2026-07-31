import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

d = json.load(open('logs/run_40ff5a370f1e472a9aadf5d8500d5424.json', 'r', encoding='utf-8'))

for e in d['tool_events']:
    tool = e['tool']
    status = e['status']
    err = str(e.get('error', ''))[:120] if e.get('error') else ''
    
    if tool == 'search_registry':
        results = e.get('result', [])
        count = len(results) if isinstance(results, list) else 0
        src = e.get('args', {}).get('source_name', '?')
        kw = e.get('args', {}).get('keyword', '***')
        print(f"  {tool} [{src}]: {status}, {count} results {err}")
    elif tool == 'verify_candidates':
        results = e.get('result', [])
        count = len(results) if isinstance(results, list) else 0
        print(f"  {tool}: {status}, {count} candidates after verify")
    elif tool == 'prepare_candidates':
        results = e.get('result', [])
        count = len(results) if isinstance(results, list) else 0
        print(f"  {tool}: {status}, {count} candidates")
    elif tool == 'enrich_candidates':
        results = e.get('result', [])
        count = len(results) if isinstance(results, list) else 0
        print(f"  {tool}: {status}, {count} candidates after enrich")
    elif tool == 'deduplicate_candidates':
        results = e.get('result', [])
        count = len(results) if isinstance(results, list) else 0
        print(f"  {tool}: {status}, {count} unique candidates")
    elif tool == 'filter_missing_metadata':
        results = e.get('result', [])
        count = len(results) if isinstance(results, list) else 0
        print(f"  {tool}: {status}, {count} candidates remaining")
    elif tool == 'rank_datasets':
        result = e.get('result')
        if isinstance(result, tuple) or isinstance(result, list):
            ranked = result[0] if result else []
            count = len(ranked) if isinstance(ranked, list) else 0
        else:
            count = 0
        print(f"  {tool}: {status}, {count} ranked {err}")
        # Print raw result type
        print(f"    result type: {type(result).__name__}, keys/len: {len(result) if isinstance(result, (list,dict)) else 'N/A'}")
    else:
        print(f"  {tool}: {status} {err}")
