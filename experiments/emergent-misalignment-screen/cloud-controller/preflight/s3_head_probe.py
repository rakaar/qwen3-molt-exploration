"""One credential-free fixed-proxy HEAD probe; no redirects or dynamic URLs."""
import http.client
import json
import ssl
import sys

result = {'host': 's3.amazonaws.com', 'method': 'HEAD', 'path': '/',
          'route': 'HTTP CONNECT via proxy:8080', 'credentials_used': False,
          'passed': False}
connection = http.client.HTTPSConnection('proxy', 8080, timeout=10,
                                       context=ssl.create_default_context())
connection.set_tunnel('s3.amazonaws.com', port=443)
try:
    connection.request('HEAD', '/')
    response = connection.getresponse()
    result.update(http_status=response.status, passed=True, redirects_followed=False)
except (OSError, http.client.HTTPException) as error:
    result.update(error_type=type(error).__name__, errno=getattr(error, 'errno', None),
                  error=str(error)[:240],
                  connect_denied=str(error).startswith('Tunnel connection failed: 403 '))
finally:
    connection.close()
print(json.dumps(result, indent=2))
sys.exit(0 if result['passed'] else 1)
