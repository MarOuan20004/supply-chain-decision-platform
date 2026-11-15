"""Gestionnaire de promotion et de chargement des modeles @champion depuis MLflow Registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from supply_chain.config import SupplyChainSettings, settings

logger = logging.getLogger(__name__)


class ChampionModelLoadError(RuntimeError):
    """Exception levee lorsque les modeles @champion ne sont pas disponibles dans MLflow."""


@dataclass(frozen=True)
class ChampionModelBundle:
    """Conteneur immuable des trois modeles charges en memoire au demarrage de FastAPI."""

    forecaster_model: Any
    classifier_model: Any
    anomaly_model: Any
    versions: dict[str, str]


def promote_latest_to_champion(cfg: SupplyChainSettings = settings) -> dict[str, str]:
    """Assigne l'alias @champion a la derniere version enregistree de chacun des 3 modeles."""
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)

    model_names = (
        cfg.registered_forecaster_name,
        cfg.registered_classifier_name,
        cfg.registered_anomaly_name,
    )
    promoted: dict[str, str] = {}

    for name in model_names:
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise ChampionModelLoadError(
                f"Aucune version enregistree trouvee pour '{name}' dans {cfg.mlflow_tracking_uri}."
            )
        latest_ver = max(versions, key=lambda mv: int(mv.version))
        client.set_registered_model_alias(
            name=name,
            alias=cfg.champion_alias,
            version=str(latest_ver.version),
        )
        promoted[name] = str(latest_ver.version)
        logger.info(
            "Modele '%s' version %s promu avec l'alias '@%s'.",
            name,
            latest_ver.version,
            cfg.champion_alias,
        )

    return promoted


def load_champion_bundle(cfg: SupplyChainSettings = settings) -> ChampionModelBundle:
    """Charge les trois modeles @champion depuis MLflow Model Registry."""
    try:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
        alias = cfg.champion_alias

        resolved_versions: dict[str, str] = {}
        for name in (
            cfg.registered_forecaster_name,
            cfg.registered_classifier_name,
            cfg.registered_anomaly_name,
        ):
            mv = client.get_model_version_by_alias(name=name, alias=alias)
            resolved_versions[name] = str(mv.version)

        forecaster = mlflow.pyfunc.load_model(f"models:/{cfg.registered_forecaster_name}@{alias}")
        classifier = mlflow.sklearn.load_model(f"models:/{cfg.registered_classifier_name}@{alias}")
        anomaly_det = mlflow.sklearn.load_model(f"models:/{cfg.registered_anomaly_name}@{alias}")

        return ChampionModelBundle(
            forecaster_model=forecaster,
            classifier_model=classifier,
            anomaly_model=anomaly_det,
            versions=resolved_versions,
        )
    except Exception as exc:
        raise ChampionModelLoadError(
            f"Impossible de charger les modeles @{cfg.champion_alias} depuis MLflow: {exc}"
        ) from exc
