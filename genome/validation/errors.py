"""Validation error types for the EAG schema layer."""


class GenomeValidationError(Exception):
    """A genome, somatic envelope, or CIG record failed validation.

    ``errors`` lists every violation found (structural + semantic).
    """

    def __init__(self, errors):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class RegistryIntegrityError(GenomeValidationError):
    """Registry content contradicts its content address (collision or tampering)."""


class RecordConsistencyError(GenomeValidationError):
    """CIG aggregate/child references are inconsistent."""
