import requests, json
h = {'Authorization': 'Bearer AGLuB4FdRbMT7DMSaEHhAGLSXsCsgDWe'}
for ep in ['/api/overview','/api/capital_utilization','/api/adaptive_factors','/api/strategy_performance','/api/system_status','/api/orders','/api/health_score']:
    try:
        r = requests.get(f'http://localhost:8080{ep}', headers=h, timeout=10)
        print(f'--- {ep} {r.status_code} ---')
        print(json.dumps(r.json(), indent=2, ensure_ascii=False)[:1500])
    except Exception as e:
        print(f'--- {ep} ERROR {e} ---')
