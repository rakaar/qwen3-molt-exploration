"""No network, credentials, CLI installation, or provider mutations."""
import unittest
from spar_vast_cloud import GuardError, check_identity, guard_mutation, validate_args, LABEL


class CloudGuardChecks(unittest.TestCase):
    def test_only_pinned_spar_identity(self):
        check_identity({"id": 710735})
        for user in ({"id": 345874}, {"id": 12}, {"id": "710735"}, {}):
            with self.assertRaises(GuardError):
                check_identity(user)

    def test_no_deletion_or_credential_endpoint_override(self):
        for args in (["destroy", "instance", "1"], ["reboot", "instance", "1"],
                     ["show", "instances", "--api-k=x"], ["show", "user", "--ur=x"],
                     ["create", "instance", "1", "--onstart-cmd", "anything"]):
            with self.assertRaises(GuardError):
                validate_args(args)

    def test_stop_only_specific_task_worker(self):
        guard_mutation(["stop", "instance", "123"], lambda _: [{"id": 123, "label": LABEL}])
        with self.assertRaises(GuardError):
            guard_mutation(["stop", "instance", "123"], lambda _: [{"id": 123, "label": "someone-elses-worker"}])

    def test_duplicate_creation_refused(self):
        args = ["create", "instance", "456", "--label", LABEL, "--disk", "50"]
        with self.assertRaises(GuardError):
            guard_mutation(args, lambda _: [{"id": 123, "label": LABEL}])

    def test_creation_requires_driver_compatibility_and_budget(self):
        args = ["create", "instance", "456", "--label", LABEL, "--disk", "50"]
        offer = {"id": 456, "cuda_max_good": 12.8, "dph_total": 0.2,
                 "inet_down_cost": 0.003, "inet_up_cost": 0.004}
        def check(candidate):
            guard_mutation(args, lambda command: [] if command[:2] == ["show", "instances"] else [candidate])
        check(offer)
        for bad in ({**offer, "cuda_max_good": 12.4}, {**offer, "dph_total": 0.41},
                    {**offer, "inet_down_cost": 1.0}):
            with self.assertRaises(GuardError):
                check(bad)


if __name__ == "__main__":
    unittest.main()
