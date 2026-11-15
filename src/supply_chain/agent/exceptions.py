"""Exceptions typees pour le registre d'outils et l'orchestrateur agentique."""

from __future__ import annotations


class AgentError(Exception):
    """Classe de base de toutes les erreurs du sous-systeme agentique."""


class UnknownToolError(AgentError):
    """Levee lorsque le LLM tente d'appeler un outil absent du registre."""


class ToolValidationError(AgentError):
    """Levee lorsque les arguments ou la sortie d'un outil echouent a la validation Pydantic v2."""


class ToolExecutionError(AgentError):
    """Levee lorsqu'une erreur metier ou technique survient durant l'execution d'un outil."""


class OrchestratorTimeoutError(AgentError):
    """Levee si la duree totale de la session depasse agent_timeout_sec."""


class MaxStepsExceededError(AgentError):
    """Levee si l'orchestrateur atteint MAX_STEPS sans produire de DecisionReport valide."""
