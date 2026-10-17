"""Controller-only compatibility for official Vast prepared requests.

The official package is unchanged. Use inherited managed proxy and CA trust,
and reject provider redirects outside the approved credential destination.
This module never reads credentials.
"""
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlsplit


def configure(session_class):
    policy = json.loads(Path('/etc/codex/network-policy.json').read_text())
    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    ca = os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')
    if not proxy or proxy != policy.get('http_proxy'):
        raise RuntimeError('Inherited managed proxy configuration is required')
    if not ca or not Path(ca).is_file():
        raise RuntimeError('Inherited configured CA bundle is required')
    original_init = session_class.__init__
    original_send = session_class.send
    original_redirect = session_class.get_redirect_target

    def restricted_init(self, server_url):
        check_destination(server_url)
        original_init(self, server_url)
        self.verify = ca
        self.proxies = {'http': proxy, 'https': proxy}

    def restricted_send(self, request, **kwargs):
        check_destination(request.url)
        # Follow only the identity endpoint's observed canonical path; retain
        # the approved origin rather than following its cross-host redirect.
        if request.url == 'https://cloud.vast.ai/api/v0/users/current':
            request.url = 'https://cloud.vast.ai/api/v0/users/current/'
        if kwargs.get('verify') is False:
            raise RuntimeError('TLS verification may not be disabled')
        kwargs['verify'] = ca
        kwargs['proxies'] = {'http': proxy, 'https': proxy}
        return original_send(self, request, **kwargs)

    def observed_redirect(self, response):
        target = original_redirect(self, response)
        if target:
            source = urlsplit(response.url)
            destination = urlsplit(urljoin(response.url, target))
            record = {'status_code': response.status_code,
                      'source': {'scheme': source.scheme, 'host': source.hostname,
                                 'path': source.path},
                      'destination': {'scheme': destination.scheme,
                                      'host': destination.hostname,
                                      'path': destination.path},
                      'query_and_fragment_omitted': True}
            path = Path('/workspace/work/preflight/last-provider-redirect.json')
            path.write_text(json.dumps(record, indent=2) + '\n')
        return target

    session_class.__init__ = restricted_init
    session_class.send = restricted_send
    session_class.get_redirect_target = observed_redirect


def check_destination(url):
    parts = urlsplit(url)
    if (parts.scheme != 'https' or parts.hostname != 'cloud.vast.ai'
            or parts.port not in (None, 443) or parts.username or parts.password):
        host = parts.hostname or '(missing)'
        raise RuntimeError('Provider destination outside approved HTTPS host refused; host=' + host)
