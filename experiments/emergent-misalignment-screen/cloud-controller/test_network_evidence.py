"""Fail closed when functional proof is stale, unscoped, or insecure."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import network_evidence as network


class NetworkEvidenceChecks(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'policy.json'
        path.write_text(json.dumps({'version': 1, 'http_proxy': network.PROXY,
            'http_network_policy': {'type': 'restricted', 'egress_rules': [{'host': 's3.amazonaws.com'}]}}))
        for mock in (patch.object(network, 'SNAPSHOT', path),
                     patch.object(network.urllib.request, 'proxy_bypass', return_value=False),
                     patch.dict(network.os.environ, {'HTTPS_PROXY': network.PROXY})):
            mock.start()
            self.addCleanup(mock.stop)
        now = datetime.now(timezone.utc).isoformat()
        self.evidence = {'observed_at_utc': now, 'observations_current': True,
            'network_policy': {'state': 'unknown', 'effective_allowed_hosts': ['s3.amazonaws.com']},
            'functional_proxy_proof': {'observed_at_utc': now, 'proxy': network.PROXY,
                'network_snapshot_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'credential_free': True,
                'allowed': {'s3.amazonaws.com': {'proxy_connect_status': 200,
                    'tls_verify_result': 0, 'redirects_followed': 0, 'exit_code': 0, 'http_status': 405}},
                'deny_control': {'hostname': 'example.org', 'proxy_connect_status': 403,
                    'http_status': 0, 'exit_code': 56, 'redirects_followed': 0}}}

    def test_unknown_requires_independent_functional_proof_and_stays_unknown(self):
        network.require_network_permission(self.evidence, 's3.amazonaws.com')
        self.assertEqual(self.evidence['network_policy']['state'], 'unknown')
        del self.evidence['functional_proxy_proof']
        with self.assertRaises((ValueError, KeyError)):
            network.require_network_permission(self.evidence, 's3.amazonaws.com')

    def test_stale_unverified_tls_redirect_and_missing_deny_are_rejected(self):
        variants = []
        stale = copy.deepcopy(self.evidence)
        stale['functional_proxy_proof']['observed_at_utc'] = (datetime.now(timezone.utc)-timedelta(minutes=6)).isoformat()
        variants.append(stale)
        for field, value in [('tls_verify_result', 1), ('redirects_followed', 1), ('exit_code', 60)]:
            bad = copy.deepcopy(self.evidence)
            bad['functional_proxy_proof']['allowed']['s3.amazonaws.com'][field] = value
            variants.append(bad)
        bad = copy.deepcopy(self.evidence)
        bad['functional_proxy_proof']['deny_control']['proxy_connect_status'] = 200
        variants.append(bad)
        for evidence in variants:
            with self.assertRaises(ValueError):
                network.require_network_permission(evidence, 's3.amazonaws.com')

    def test_changed_policy_or_different_host_is_rejected(self):
        with self.assertRaises(ValueError):
            network.require_network_permission(self.evidence, 'other.example')
        network.SNAPSHOT.write_text(network.SNAPSHOT.read_text() + '\n')
        with self.assertRaises(ValueError):
            network.require_network_permission(self.evidence, 's3.amazonaws.com')

    def test_other_platform_states_and_noncurrent_observations_are_rejected(self):
        for state in ('unsupported', 'skipped', None):
            bad = copy.deepcopy(self.evidence)
            bad['network_policy']['state'] = state
            with self.assertRaises(ValueError):
                network.require_network_permission(bad, 's3.amazonaws.com')
        self.evidence['observations_current'] = False
        with self.assertRaises(ValueError):
            network.require_network_permission(self.evidence, 's3.amazonaws.com')


if __name__ == '__main__':
    unittest.main()
